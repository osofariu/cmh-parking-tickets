# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Exploring CMH Parking Violations

# ## Imports
#
# * pandas, numpy
#     * manipulate all the things
# * matplotlib, seaborn
#     * histograms, pairplots, etc.
# * osmnx
#     * display violation location on a map
# * networkx
#     * calculate shortest path
#     * set-up graph for mapping a graph with heatmap attributes
# * folium
#     * example of interactive maps

# Edit notebook in external editor, so don't auto-save from jupyter notebook
# this is a trial.. looking for a good way to preserve diffs
# %autosave 0

# +
# #%matplotlib inline

import datetime
import hashlib
import math


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import osmnx as ox
import networkx as nx

import folium
from IPython.display import IFrame

# -

# ## Loading data

# ### Violations

# [Columbus City Parking Violations and Ticket Status 2013-2018 -- COLUEXTRACT](
# https://discovery.smartcolumbusos.com/dataset/conduent/160c98a1_ad56_4658_8553_5ee8e7d0d953) - Smart Columbus
#
# This dataset covers the parking violations identified by Parking Enforcement Officer (PEO) and the tickets issued
# for those violations. Also, the data identifies the status of the ticket (e.g., paid, dismissed, etc.,)
# This dataset covers years 2013 through 2018.

violations = pd.read_csv("../data/raw/160c98a1-ad56-4658-8553-5ee8e7d0d953.csv.gz",
                         compression='gzip',
                         low_memory=False)

pd.set_option('display.max_columns', None)
violations.info()

# ### Dictionary for parking violations
# An explanation of all the fields in the violations dataset
#

violations_dictionary = pd.read_json(
    "https://data.smartcolumbusos.com/api/v1/dataset/160c98a1-ad56-4658-8553-5ee8e7d0d953/dictionary")
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', -1)
violations_dictionary.head(98)

# ### Meters
#

# Meters is a dataset that contains the location of every meter in Columbus.  The Violations dataset reports the
# meter where a violation was issued, but the location is often missing, and even when it's there it's location
# doesn't precisely match that of the meter for which the violation was issued.  I suspect this is because the
# location is based on where the the Agent (or their hand-held device) when the data was transmitted.
#
# Thankfully the meter number is present in the Violations dataset, so we can join the Violations and Meters dataset
# to get the location of the meter for which a violation was issued.

meters = pd.read_csv("../data/raw/d9b11b8f-67f3-48c4-8831-0f22d93166ce")

# ## Extract interesting data from Violations dataset

# work with the most promising (on first pass) columns:
violations_columns_use = ['ticket', 'entity', 'meter', 'iss dt', 'due', 'fine', 'make', 'iss time',
                          'badge', 'pay amt', 'lat', 'long']

# extract date-related fields
violations['issue_date'] = violations['iss dt'].map(lambda dt: datetime.datetime.strptime(str(dt), '%Y%j'))
violations['year'] = violations['issue_date'].map(lambda dt: dt.year)
violations['dayOfWeek'] = violations['issue_date'].map(lambda dt: dt.weekday())

violations['hour'] = np.floor(violations['iss time'] / 100)

# convert 'pay amt' and 'fine' into float datatype
trans_dict = {ord('$'): None, ord('('): None, ord(')'): None, ord(","): None}
violations['amt_float'] = violations['pay amt'].str.translate(trans_dict).astype(float)
violations['fine_float'] = violations['fine'].str.translate(trans_dict).astype(float)

# let's see the data
computed_columns = ['issue_date', 'year', 'dayOfWeek', 'hour', 'amt_float', 'fine_float']
violations[violations_columns_use + computed_columns].head(10)

# date columns
print(violations[['iss dt', 'iss time', 'issue_date', 'year', 'dayOfWeek', 'hour']])

# ## About fines

# #### Show distribution of fine and pay_amt

# +
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
sns.distplot(violations[violations['amt_float'] < 100]['amt_float'], bins=20, ax=ax1)
sns.distplot(violations[(violations['amt_float'] > 100) & (violations['amt_float'] <= 200)]['amt_float'], bins=20,
             ax=ax2)
sns.distplot(violations[(violations['amt_float'] > 200) & (violations['amt_float'] <= 600)]['amt_float'], bins=20,
             ax=ax3)

# TODO - show the numer of samples on each graph.. maybe also add another graph that shows all the data
# currently you can't tell what the overall distribution of fine amounts is.
# -

# this is an outlyer..
# TODO - pay amount currently strips out the negative, but the example below amy be a credit.  We might
# want to parse the amount so that negative amouts show up.. or alternately filter them out.
larger_amt = violations[(violations['amt_float'] >= 600)]
print(larger_amt)

_, ax0 = plt.subplots(1, 1, figsize=(20, 5))
sns.distplot(violations['fine_float'], bins=100, ax=ax0)

# +
_, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

sns.distplot(violations[violations['fine_float'] < 100]['fine_float'], bins=20, ax=ax1)
sns.distplot(violations[(violations['fine_float'] >= 100) & (violations['fine_float'] < 200)]['fine_float'], bins=20,
             ax=ax2)
sns.distplot(violations[(violations['fine_float'] >= 200) & (violations['fine_float'] < 600)]['fine_float'], ax=ax3)
# # TODO - show the numer of samples on each graph (same as for the pay amount graphs)

# +
violations.info()

# note that the data below doesn't show correctly on GitHub, but looks fine in the notebook
violations.sample(10)
# -

# aggregate pay amount and fine by entity (who received the fine)
entity_tickets = violations[['entity', 'amt_float', 'fine_float']].groupby('entity').agg(['count', 'sum']).sort_values(
    [('amt_float', 'count')], ascending=False)
# distribution of outliers - entities who got more than 25 tickets
print(entity_tickets.head(25))

# #### distribution of violations over time

# wonder why number of violations issued went down the last three years?
violations['year'].value_counts()

# #### Distribution of fines by day of the week

# Monday and Sunday are lighter days for parking tickets
sns.distplot(violations['dayOfWeek'], bins=7, axlabel="Mon .. Sun")

# #### Distribution of fines by time of the day

plt.figure(figsize=(15, 6))
ax = sns.distplot(violations['iss time'], bins=24, norm_hist=True, )
ax.minorticks_on()

# #### All about agents (as identified by badge) who gave most tickets

# +
print(violations[['badge']].describe())
agent_business = violations[['badge', 'year', 'entity', 'ticket', 'fine_float']] \
    .groupby(['badge', 'year']) \
    .agg({'fine_float': ['mean', 'min', 'max'], 'entity': ['count']})

agent_business.columns = ['fine_mean', 'fine_min', 'fine_max', 'entity_count']
agent_business.reset_index().sort_values('entity_count', ascending=False).head(25)

# +
violations_clean = violations[['iss time', 'fine_float', 'dayOfWeek', 'hour', 'year']].dropna()

# I don't think this tells much of a story, but keeping the pairplot as an example.. maybe
# someone will see something in this data.

sns.pairplot(violations_clean)
# -

# #### How many tickets are generally given each day?

# +
# get network graph for Columbus from disk if available (much faster to load)
try:
    G = ox.load_graphml('cmh_network.graphml')
except IOError:
    print("Download and save network.")
    G = ox.graph_from_place('Columbus, Ohio, USA', network_type='drive', simplify=False)
    ox.save_graphml(G, filename='cmh_network.graphml')
    G = ox.load_graphml('cmh_network.graphml')

fig, ax = ox.plot_graph(G)

# +
# Get min, max for lat long as coordinates from G, to get bounds for columbus
# We will use these bounds to clean up the lat/long in the violations data, which is not very clean
cmh_gdf = ox.graph_to_gdfs(G)
print(cmh_gdf[0].x.agg(['min', 'max']))
print(cmh_gdf[0].y.agg(['min', 'max']))

cmh_n = float(cmh_gdf[0].y.agg('max'))
cmh_s = float(cmh_gdf[0].y.agg('min'))
cmh_e = float(cmh_gdf[0].x.agg('max'))
cmh_w = float(cmh_gdf[0].x.agg('min'))

print(cmh_n, cmh_s, cmh_e, cmh_w)
# -

# convert lat/long to usable lat/long float values
violations['lat_float'] = violations['lat'] / 1000000
violations['long_float'] = violations['long'] / 1000000 * -1

# #### Exploring Meters

# +
# meters

meters.info()

# +
# meters[meters['meter number'] == 'EN529']

# We're joining the meters and violations data frames on column 'Meter'.
# so we will rename this column to be able to do the merge
# TODO: figure out why I can't rename this to 'meter' - it forces capitalization on this data frame
meters = meters.rename(columns={'Meter Number': 'Meter'})
meters.info()
# -

# ### Merging violations and meters

# +
violations = violations.rename(columns={'meter': 'Meter'})
violations.info()
meters_loc = meters[['Meter', 'Lat', 'Long']]

violations_x_meters = violations.merge(meters, how='left', on='Meter')
# -

# #### Set the violation location from the meters dataset if available

# TODO: make faster; Dan suggested using a WHERE cause which takes advantage of vectorization
fudge = 0.2
violations_x_meters['lat_combined'] = violations_x_meters \
    .apply(lambda p: p['Lat']
if (p['Lat'] > cmh_s - fudge) & (p['Lat'] < cmh_n + fudge)
else p['lat_float'], axis=1)

violations_x_meters['long_combined'] = violations_x_meters \
    .apply(lambda p: p['Long']
if (p['Long'] > cmh_w - fudge) & (p['Long'] < cmh_e + fudge)
else p['long_float'], axis=1)

violations_x_meters.info()

# only keep the records with an actual location
violations_x_meters = violations_x_meters[
    (violations_x_meters['lat_combined'] > 0) & (violations_x_meters['long_combined'] < 0)]
print(violations_x_meters.shape)

# #### Distribution of number of violations given per day in Columbus

violations_by_day = violations[['issue_date', 'Meter']].groupby(['issue_date']).count()
sns.distplot(violations_by_day)
print(violations_by_day)

# This if the final data that we will be plotting on a map
violations_x_meters.head(12)

_, ax_violations = ox.plot_graph(G, fig_height=15, show=False, close=False)
ax_violations.scatter(violations_x_meters['long_combined'], violations_x_meters['lat_combined'], alpha=0.5, c='red')

# +
# Pick a random data set that's somewhat clustered, to create a heat map.

heat_tickets = violations_x_meters[(violations_x_meters['badge'] == '19') &
                                   (violations_x_meters['issue_date'] == '2014-02-13')] \
    .sort_values(by='iss time') \
    [['Meter', 'lat_combined', 'long_combined']].dropna()

print(heat_tickets.shape)

# +
# Would like to create a heat map for just this area, so that we can zoom into it on the map
# initially I plotted the heat map for these 63 points on the entire map of columbus..
# and the highlighted area was too small.
bbox = ox.core.bbox_from_point((39.966358, -83.001397), 2500)
north, south, east, west = bbox

G_small = ox.graph_from_bbox(north, south, east, west, simplify=True, retain_all=False)
_, ax_small_violations = ox.plot_graph(G_small, fig_height=15, show=False, close=False)
ax_small_violations.scatter(heat_tickets['long_combined'], heat_tickets['lat_combined'], alpha=0.5, c='red')


# +
# get nearest node (on the OSMNX graph) to each meter location; we can't just add these points
# to the Open Street map without connecting them to the existing graph; perhaps we can do something
# more clever there to get better accuracy on these locations.  We did set "simplify=false" when
# importing the original map of Columbus to hopefully get more points near meters.

def heat_nodes(heat_tickets):
    for ticket in heat_tickets.values:
        _, lat, long = ticket
        yield ox.get_nearest_node(G_small, (lat, long))


# +
# temporarily bringing in heatmapx and making it work with OpenStreet maps until we can make
# the package support our use case

# # %load ~/Library/Caches/pypoetry/virtualenvs/cmh-packing-violations-JzEtr0-_-py3.7/lib/python3.7/site-packages/heatmapx/__init__.py
__version__ = '0.1.0'
__all__ = ['temperature_graph']

import itertools

from typing import Iterable, Optional, Union


def temperature_graph(
        G: nx.Graph,
        source_nodes: Iterable,
        depth_limit: Optional[int] = None,
        heat_increments: Union[Iterable, float] = 1,
        heat_key: str = 'heat'
) -> nx.Graph:
    """
    Calculate temperatures radiating from heat sources in a graph.

    Temperature values are initially set to 0 and then updated throughout `G` in
    a breadth-first manner beginning at each node in `source_nodes`.  For each
    source node `s`, the temperature of each edge `e` and its incident nodes in
    `G` are updated according to `heat_increments` and how many edges away they
    are from the source node `s`.  This process is repeated for every source
    node, with temperatures from multiple source nodes contibuting additively to
    the overall temperature of the nodes and edges in the graph.


    Parameters
    ----------
    G : networkx.Graph
        The graph from which to generate a heatmap.  A copy of the graph will be
        produced by default.

    source_nodes : Iterable
        The nodes serving as heat sources in `G`.

    depth_limit : Optional[int]
        The maximum number of edges away from a source node to update
        temperature values.  (Default: 0)

    heat_increments : Union[Iterable, float]
        A sequence whose `n`-th element gives, for each source node `s`, the
        amount to update the temperature of each node and edge that is `n`
        breadth-first layers away from `s`.  A constant value may also be
        provided to apply to all nodes and edges in the same connected component
        as each source node.  (Default: 1)

    heat_key : str
        The name of the node and edge attribute where temperature values will be
        stored in `T`.

    Returns
    -------
    T : networkx.Graph
        A copy of `G` in which every node and edge has its temperature stored in
        a `heat_key` attribute.
    """
    T = type(G)()
    T.add_nodes_from(G.nodes(data=True), **{heat_key: 0})
    T.add_edges_from(G.edges(data=True), **{heat_key: 0})

    try:
        heat_increments = iter(heat_increments)
    except TypeError:
        heat_increments = itertools.repeat(heat_increments)

    for source in source_nodes:
        visited_nodes = set()
        data_by_depth = itertools.islice(
            zip(_edge_bfs_by_depth(T, [source]), heat_increments),
            depth_limit)
        for edges_at_depth, heat_increment in data_by_depth:
            for edge in edges_at_depth:
                _update_edge_temperature(T, edge, heat_key, heat_increment)
                _update_node_temperatures(T, edge, heat_key, heat_increment,
                                          visited_nodes)
    return T


def _edge_bfs_by_depth(G, source_nodes, orientation=None):
    yield from _group_by_sources(
        nx.edge_bfs(G, source_nodes, orientation),
        set(source_nodes))


def _group_by_sources(edges_iterator, initial_sources):
    edges = iter(edges_iterator)

    current_group_sources = set(initial_sources)
    current_group = []

    for current_edge in edges:
        if current_edge[0] in current_group_sources:
            current_group.append(current_edge)
        else:
            yield current_group
            current_group_sources = {target for _, target, _ in current_group}
            current_group = [current_edge]

    yield current_group


def _update_edge_temperature(G, edge, heat_key, heat_increment):
    G.edges[edge][heat_key] += heat_increment


def _update_node_temperatures(G, nodes, heat_key, heat_increment, visited_nodes):
    #     print(G, nodes, heat_key, heat_increment, visited_nodes)
    for node in set(nodes[:2]).difference(visited_nodes):
        G.nodes[node][heat_key] += heat_increment
        visited_nodes.add(node)


# -

# calculate the heat of each node; this adds a 'heat' attribute to each node in the graph
# the more near connections to other 'hot' nodes the higher the heat value
T = temperature_graph(G_small, heat_nodes(heat_tickets), heat_increments=1, depth_limit=10)

# # copy node attributes from G_small
nx.set_node_attributes(T, {node: data for node, data in G_small.nodes(data=True)})

# # copy edge attributes from G_small
nx.set_edge_attributes(
    T, {(source, target, key): data
        for source, target, key, data
        in G_small.edges(keys=True, data=True)})

# OSMNX wants these set in order to be able to graph T
T.graph['crs'] = G_small.graph['crs']
T.graph['name'] = G_small.graph['name']

# extract node and edge heat values, so we can map each value to a different color on the grapgh
heat_node_values = [heat for (id, heat) in list(T.nodes(data='heat'))]
heat_edge_values = [heat for (source, dest, heat) in list(T.edges(data='heat'))]
print(len(heat_node_values), len(heat_edge_values))

# +
map_heat_to_color = {
    0: "#f8f7f6", 1: "#fff5ef", 2: "#ffe7dc", 3: "#ffdac8", 4: "#ffcdb4",
    5: "#ffc0a1", 6: "#ffc0a1", 7: "#ffb38d", 8: "#ffa67a", 9: "#ff9966",
    10: "#ff8c52", 11: "#ff7f3f", 12: "#ff722b", 13: "#ff6518", 14: "#ff5804",
    15: "#ef5000", 16: "#dc4900", 17: "#c84300", 18: "#b43c00", 19: "#a13600",
    20: "#8d2f00", 21: "#7a2900", 22: "#662200", 23: "#662200", 24: "#662200"}

# TODO: test this function!  Added pytest to the project, but need to extract code that we need
# to test into separate files so we can test them.
# basically when we have more heat values than we have colors in our map (we have 25 colors), then
# we pick an existing color by applying a ratio..
def heat_to_color(index, max_index):
    use_index = index
    if max_index >= len(map_heat_to_color):
        reduce_ratio = len(map_heat_to_color) / (max_index + 1)
        use_index = math.floor(index * reduce_ratio)
    return map_heat_to_color[use_index]


# -

node_heat_colors = [heat_to_color(heat, max(heat_node_values)) for heat in heat_node_values]
edge_heat_colors = [heat_to_color(heat, max(heat_edge_values)) for heat in heat_edge_values]
print(len(node_heat_colors), len(edge_heat_colors))

# noinspection PyTypeChecker
ox.plot_graph(T, fig_height=15, show=False, close=False, node_color=node_heat_colors, edge_color=edge_heat_colors)

# ### Plot a route from two points on the graph
#
# This section is just used as a demonstration for now.

tickets = heat_tickets
origin_point = (tickets[0:1].lat_combined.iloc[0], tickets[0:1].long_combined.iloc[0])
destination_point = (tickets[61:62].lat_combined.iloc[0], tickets[61:62].long_combined.iloc[0])

origin_node = ox.get_nearest_node(G, origin_point)
destination_node = ox.get_nearest_node(G, destination_point)
print(origin_node, destination_node)

# +
# TODO: make this useful; for example, plot a route between every visited parking meter, ordered by time 
#to see the most likely route taken.  

#To start with we are plotting a route between only the first and last visited points, to get a hang 
# of how to plot routes.

bbox = ox.core.bbox_from_point((39.966358, -83.001397), 2500)
north, south, east, west = bbox

G_small_for_route = ox.graph_from_bbox(north, south, east, west, 
                                       network_type='drive', simplify=False, retain_all=True)

route = nx.shortest_path(G_small_for_route, origin_node, destination_node, weight='length')

# when plotting a route from a bounded box it fails.. something gets extracted, but it works with full map
# projected = ox.project_graph(G_small)
ox.plot_graph_route(G_small_for_route, route, origin_point=origin_point,
                    destination_point=destination_point, fig_height=15, show=False, close=False)

# -

# ### Color-code tickets by issuing Agent

# +
# Plot violations given on a single day, 2013-01-03, color-coded by the individual issuing agent
# TODO: This could be expanded to compare different agents by day or time

violations_for_day = violations_x_meters[violations_x_meters['issue_date'] == '2013-01-03']

bbox = ox.core.bbox_from_point((39.977110, -83.003500), 4000)
north, south, east, west = bbox
G0 = ox.graph_from_bbox(north, south, east, west, network_type='drive_service')
_, ax_day_violations = ox.plot_graph(G0, fig_height=15, show=False, close=False)

# badge number is a string; this function maps that to an integer value
# TODO: test this function!
def encode_badge(badge):
    encoded_badge = str(badge).encode('utf-8')
    return int(hashlib.sha1(encoded_badge).hexdigest(), 16) % (10 ** 8) % 10000

# list of colors for each violation, where an individual badge gets its own color
# we pass this list to the `scatter` function to it can pick the right color for each point
badge_to_colors = violations_for_day['badge'].map(encode_badge)
ax_day_violations.scatter(violations_for_day['long_combined'], violations_for_day['lat_combined'],
                          alpha=0.5, c=badge_to_colors, marker='h')
# -

bbox = ox.core.bbox_from_point((39.977110, -83.003500), 500)
north, south, east, west = bbox
short_north_violations = violations_x_meters[(violations_x_meters['long_combined'] < east) &
                                             (violations_x_meters['long_combined'] > west) &
                                             (violations_x_meters['lat_combined'] < north) &
                                             (violations_x_meters['lat_combined'] > south) &
                                             (violations_x_meters['year'] == 2017)
                                             ]
print(short_north_violations.shape)
G1 = ox.graph_from_bbox(north, south, east, west, network_type='drive_service')

print(short_north_violations.shape)

early_morning_violations = short_north_violations[(short_north_violations['iss time'] >= 500) & \
                                                  (short_north_violations['iss time'] < 700)]
morning_violations = short_north_violations[(short_north_violations['iss time'] >= 700) & \
                                            (short_north_violations['iss time'] < 1000)]
midday_violations = short_north_violations[(short_north_violations['iss time'] >= 1000) & \
                                           (short_north_violations['iss time'] < 1400)]
afternoon_violations = short_north_violations[(short_north_violations['iss time'] >= 1400) & \
                                              (short_north_violations['iss time'] < 1800)]
evening_violations = short_north_violations[(short_north_violations['iss time'] >= 1800) & \
                                            (short_north_violations['iss time'] < 2200)]
night_violations = short_north_violations[(short_north_violations['iss time'] >= 2300) & \
                                          (short_north_violations['iss time'] < 500)]

# #### Create bounding box 1 mile around the Forge, and plot violations around this area
#
# The point here is to see how violations vary by time for the day, for the same area.

# +
# TODO: The colors for each `scatter` graph overlap on top of each other, so it's hard to see
# which area favors violations at a particular time.  It's been suggested that we stack these
# points as vertical projections, showing them on a third coordinate.

_, ax_time_of_day = ox.plot_graph(G1, fig_height=10, show=False, close=False)
# ax.scatter(violations_x_meters['long_combined'], violations_x_meters['lat_combined'], alpha=0.2, c='red')

# ax.scatter(midday_violations['long_combined'], midday_violations['lat_combined'], alpha=0.1, c='green', marker="o")
ax_time_of_day.scatter(morning_violations['long_combined'], morning_violations['lat_combined'],
                       alpha=0.1, c='red', marker="^")
ax_time_of_day.scatter(afternoon_violations['long_combined'], afternoon_violations['lat_combined'],
                       alpha=0.1, c='brown')
ax_time_of_day.scatter(evening_violations['long_combined'], evening_violations['lat_combined'],
                       alpha=0.1, c='yellow', marker="v")
# ax.scatter(night_violations['long_combined'], night_violations['lat_combined'], alpha=0.4, c='gray')

# this is the location of the Forge
ax_time_of_day.scatter([-83.003500], [39.977110], c='blue')
plt.tight_layout()

# -

# ### Hot meters -- meters that raised the highest revenue

grp = violations_x_meters[['Meter', 'amt_float']].dropna() \
    .groupby(['Meter'], as_index=False) \
    .agg(['sum']) \
    .sort_values([('amt_float', 'sum')], ascending=False) \
    .head(100)
sns.heatmap(grp)

hot_meter_numbers = pd.Series(grp.index)
hot_meters = short_north_violations[violations_x_meters['Meter'].isin(hot_meter_numbers)]
# TODO: fix warning

# +
# Mark the hottest meters (as defined above) around the Forge. We add them as markers on
# a folium map.

# Note: this is done in Javascript, and it's slow.  You don't want to do this with thousands or more points

# TODO: you can set each icon dynamically, but I had trouble choosing a different one; I trie both 
# font-awesome and glyphicon icons, and it wouldn't pick the icon I wanted.  Figure out how to change that.

# TODO: Find something more interesting to plot.. like meters that got least tickets?
route_map = ox.plot_graph_folium(G1, popup_attribute='name', edge_width=2)

folium.Marker(
    location=[39.977110, -83.003500],
    popup='The Forge',
    icon=folium.Icon(icon='glyphicon-scale', prefix='glyphicon')
).add_to(route_map)

hot_meters.apply(lambda row:
                 folium.Marker(
                     location=[row['lat_combined'], row['long_combined']],
                     popup=grp.loc[row['Meter']][('amt_float', 'sum')],
                     icon=folium.Icon(icon='scale', color='darkgreen')
                 ).add_to(route_map),
                 axis=1)

# save to disk
filepath = 'data/graph.html'
route_map.save(filepath)
# -


IFrame(filepath, width=600, height=700)
