#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import xml.etree.ElementTree as ET
import networkx as nx

graphs_dict = {}

cwd = os.getcwd()
graphes_folder = cwd + '\\' + os.path.pardir + '\\' + os.path.pardir + '\\' + "data\MoleculesClassification\gxl\\"


def get_graphs():
    for file in os.listdir(graphes_folder):
        molecule_id = file[:-4]
        new_graph = create_graph(os.path.join(graphes_folder, file))
        graphs_dict[molecule_id] = new_graph

    return graphs_dict

def create_graph(filename):

    tree = ET.parse(filename)
    g = nx.Graph()

    for node in tree.findall(".//node"):
        node_id = node.get('id')
        attribute_dict = {}
        for attribute in node.getchildren():
            attribute_name = attribute.get('name')
            #print(attribute_name)
            attribute_values = {}
            attribute_dict[attribute_name] = attribute_values
            #print(attribute_values)
            for attributeVal in attribute.getchildren():
                attribute_values[attributeVal.tag] = attributeVal.text.strip()
                #print(attributeVal.text.strip())

        g.add_node(node_id, attr=attribute_dict)

    for edge in tree.findall(".//edge"):
        edge_start = edge.get('from')
        edge_end = edge.get('to')

        attribute_dict = {}
        for attribute in edge.getchildren():
            attribute_name = attribute.get('name')
            #print(attribute_name)
            attribute_values = {}
            attribute_dict[attribute_name] = attribute_values
            #print(attribute_values)
            for attributeVal in attribute.getchildren():
                attribute_values[attributeVal.tag] = attributeVal.text.strip()
                #print(attributeVal.text.strip())

        #print(attribute_dict)
        g.add_edge(edge_start, edge_end, attr=attribute_dict)

    return g
