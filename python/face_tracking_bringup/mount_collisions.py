# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Exclude internal contacts in the rigid tool assembly from motion checks."""

from itertools import combinations
import xml.etree.ElementTree as ET


def allow_fixed_mount_collisions(robot, semantic, *, flange_link='link6'):
    """
    Update SRDF only for collision bodies fixed to the tool flange in URDF.

    Follow fixed joints downstream, never across a moving wrist joint. Frames
    without geometry still connect the assembly when the monitor is omitted.
    Existing SRDF exclusions are preserved without adding duplicate pairs.
    """
    links = {link.get('name'): link for link in robot.findall('link')}
    if flange_link not in links:
        raise ValueError(f'Tool flange link is missing: {flange_link}')
    children = {}
    for joint in robot.findall('joint'):
        if joint.get('type') == 'fixed':
            parent = joint.find('parent').get('link')
            children.setdefault(parent, []).append(joint.find('child').get('link'))
    rigid = set()
    pending = [flange_link]
    while pending:
        name = pending.pop()
        if name not in rigid:
            rigid.add(name)
            pending.extend(children.get(name, ()))
    bodies = sorted(name for name in rigid if links[name].find('collision') is not None)
    existing = {frozenset((pair.get('link1'), pair.get('link2')))
                for pair in semantic.findall('disable_collisions')}
    for first, second in combinations(bodies, 2):
        if frozenset((first, second)) not in existing:
            ET.SubElement(semantic, 'disable_collisions', {
                'link1': first, 'link2': second, 'reason': 'Fixed'})
