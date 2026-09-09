# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Prepare the existing table world for real rendered face observations."""

import xml.etree.ElementTree as ET


def write_vision_world(base_world, destination, people_count=1):
    tree = ET.parse(base_world)
    world = tree.getroot().find('world')
    plugin = ET.SubElement(world, 'plugin', {
        'filename': 'gz-sim-sensors-system', 'name': 'gz::sim::systems::Sensors'})
    ET.SubElement(plugin, 'render_engine').text = 'ogre2'
    # Front illumination and sufficient ambient light for the existing face texture.
    world.find('scene/ambient').text = '0.65 0.65 0.65 1'
    world.find('scene/shadows').text = 'false'
    world.find("light[@name='sun']/direction").text = '0.5 0 -1'
    for index in range(people_count):
        include = ET.SubElement(world, 'include')
        ET.SubElement(include, 'uri').text = 'model://person_standing'
        ET.SubElement(include, 'name').text = 'face_test_person' + ('_2' if index else '')
        ET.SubElement(include, 'static').text = 'true'
        ET.SubElement(include, 'pose').text = '0 0 -10 0 0 0'
    tree.write(destination, encoding='utf-8', xml_declaration=True)
