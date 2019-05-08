#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#


import os
import xml.etree.ElementTree as ET
import networkx as nx

data_path = 'C:/Users/Quentin.Meteier/Documents/Cours Uni/Pattern Recognition/Repo/PatternRecognition_JungleSpeed/data/MoleculesClassification/gxl'
graphs_dict = {}


def get_graphs():
    for file in os.listdir(data_path):
        molecule_id = file[:-4]
        new_graph = create_graph(os.path.join(data_path, file))
        graphs_dict[molecule_id] = new_graph

    return graphs_dict

    # cwd = os.getcwd()
    # data_folder = cwd + "/data/gxl/"
    # file0 = "16.gxl"
    # file1 = "35.gxl"
    # g1 = create_graph(data_folder + file0)
    # molecule_id = file0[:-4]
    # print("Creating graph for molecule with ID " + molecule_id)
    # g2 = create_graph(data_folder + file1)
    # molecule_id = file0[:-4]
    # print("Creating graph for molecule with ID " + molecule_id)
    #
    # result = ged.compare(g1, g2)
    # print ("Normalized graph edit distance = %s" % result)


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
