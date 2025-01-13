from builtins import staticmethod, enumerate
from typing import Union
import numpy


def _areas_belong_to_the_same_columns(p1: list, p2: list, guess1: dict, guess2: dict, margin):
    width1 = p1[2] - p1[0]
    width2 = p2[2] - p2[0]
    center1 = (p1[0] + p1[2]) / 2
    center2 = (p2[0] + p2[2]) / 2
    guess_center1 = (guess1["guess_left"] + guess1["guess_right"]) / 2
    guess_center2 = (guess2["guess_left"] + guess2["guess_right"]) / 2
    ret = False
    if (abs(p1[0] - p2[0]) - margin < 0 or abs(guess1["guess_left"] - guess2["guess_left"]) - margin < 0
            or abs(center1 - center2) - margin < 0 or abs(guess_center1 - guess_center2) - margin < 0
            or abs(center1 - guess_center2) - margin < 0 or abs(guess_center1 - center2) - margin < 0):
        ret = abs(width1 - width2) - margin < 0
        if not ret:
            ret = abs(guess1["guess_width"] - guess2["guess_width"]) - margin < 0
        if not ret:
            ret = abs(guess1["guess_width"] - width2) - margin < 0
        if not ret:
            ret = abs(width1 - guess2["guess_width"]) - margin < 0
    return ret


def calculate_coordinates(p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
    """
    Calculate coordinate values with different type of parameters
    :param p1: Can be an int,  a list, or zero. If its type is int, represents the value of x1 (left border). If its
    type is a list, represents a set of box. If the list has 2 positions represents the left/top vertex. if the
    list has 4 positions represents the vertex of the diagonal of a rectangular area as x1, y1, x2, y2.
    :param p2: Can be an int,  a list, or zero. If its type is int, represents the value of y1 (left border). If its
    type is a list, represents the box of the right/bottom vertex of the rectangular area.
    :param x2: represents x2 coordinate of the rectangular area (right border)
    :param y2: represents y2 coordinate of the rectangular area (bottom border)
    :return: 4 integer values representing x1, y1, x2, y2
    """
    if (type(p1) is int or type(p1) is numpy.int64) and (type(p2) is int or type(p2) is numpy.int64):
        x1 = p1
        y1 = p2

    if type(p2) is list or type(p2) is tuple or type(p2) is numpy.ndarray:
        x2, y2 = p2
        x2, y2, _, _ = calculate_coordinates(x2, y2)

    if type(p1) is list or type(p1) is tuple or type(p1) is numpy.ndarray:
        if 4 == len(p1):
            x1, y1, x2, y2 = p1
            x1, y1, x2, y2 = calculate_coordinates(x1, y1, x2, y2)
        else:
            x1, y1 = p1
            x1, y1, _, _ = calculate_coordinates(x1, y1)

    return x1, y1, x2, y2


def horizontal_overlapping_ratio(box1: list, box2: list, threshold: int):
    margin = threshold + threshold
    if len(box1) == 4:
        b1x = [box1[0], box1[2]]
    else:
        b1x = box1

    if len(box2) == 4:
        b2x = [box2[0], box2[2]]
    else:
        b2x = box2
    iwidth = min(b1x[1], b2x[1]) - max(b1x[0], b2x[0])
    return iwidth / (b1x[1] - b1x[0] - margin)


def overlap_horizontally(box1: list, box2: list, threshold: int):
    """
    Check if 2 rectangular areas intersect horizontally or not
    :param box1: represents the box of a rectangular area as a list of integers. If the list has only 2 positions
    the values are exclusively of vertical box (y1, y2). If has 4 positions represents teh 4 box (x1, y1, x2, y2)
    :param box2: represents the box of a rectangular area as a list of integers. If the list has only 2 positions
    the values are exclusively of vertical box (y1, y2). If has 4 positions represents teh 4 box (x1, y1, x2, y2)
    :param threshold: represents the margin of error in absolute value to check the overlapping.
    :return: true if the boxes intersect, false otherwise.
    """
    if len(box1) == 4:
        b1y = [box1[0], box1[2]]
    else:
        b1y = box1

    if len(box2) == 4:
        b2y = [box2[0], box2[2]]
    else:
        b2y = box2

    dif = min(b1y[1], b2y[1]) - max(b1y[0], b2y[0]) - threshold
    return dif > 0


def overlap_vertically(box1: list, box2: list, threshold: int):
    """
    Check if 2 rectangular areas intersect vertically or not
    :param box1: represents the box of a rectangular area as a list of integers. If the list has only 2 positions
    the values are exclusively of vertical box (y1, y2). If has 4 positions represents teh 4 box (x1, y1, x2, y2)
    :param box2: represents the box of a rectangular area as a list of integers. If the list has only 2 positions
    the values are exclusively of vertical box (y1, y2). If has 4 positions represents teh 4 box (x1, y1, x2, y2)
    :param threshold: represents the margin of error in absolute value to check the overlapping.
    :return: true if the boxes intersect, false otherwise.
    """
    if len(box1) == 4:
        b1y = [box1[1], box1[3]]
    else:
        b1y = box1

    if len(box2) == 4:
        b2y = [box2[1], box2[3]]
    else:
        b2y = box2

    dif = min(b1y[1], b2y[1]) - max(b1y[0], b2y[0]) - threshold
    return dif > 0


def contains(edges_in_account: list, container: list, box: list, threshold, limit_values=None):
    """
    Check if a rectangular area contains completely other one. The parameter edges_in_account allows to dismiss the check
    of some coordinates. In any case, Even if a coordinate is dismissed, the box to be checked should never exceed
    the maximum limit received through the parameter limit_values
    :param edges_in_account: Coordinates to take int account in the checking. Is a list that can take the values 0, 1,
    2 or 3 or any combination of them.
    :param container: Is a list of the coordinates of the container area
    :param box:Is a list with the coordinates of the box to check
    :param threshold: represents the margin of error in absolute value to check.
    :param limit_values: represents the limite values of the container in case that some edge isn't checked
    :return: True if the container contains the box o false otherwise.
    """
    ret = False
    for i in range(4):
        dif = 0
        if i < 2:
            if i in edges_in_account:
                dif = box[i] - container[i] + threshold
            elif limit_values is not None:
                dif = box[i] - limit_values[i] + threshold
        else:
            if i in edges_in_account:
                dif = container[i] - box[i] + threshold
            elif limit_values is not None:
                dif = limit_values[i] - box[i] + threshold
        ret = dif >= 0
        if not ret:
            break
    return ret


class ThresholdAttribute:
    default_threshold = 20

    def __init__(self):
        self._threshold = ThresholdAttribute.default_threshold

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, v):
        self._threshold = v


class AbstractSection:

    def __init__(self, boundaries_or_proposed_section: Union[dict, list, numpy.ndarray] = None,
                 threshold: Union[int, ThresholdAttribute] = None, container=None, proposed_section=None,
                 boundaries=None, is_bottom_expandable=None):
        if proposed_section is None and type(boundaries_or_proposed_section) is dict:
            proposed_section = boundaries_or_proposed_section
        if boundaries is None and boundaries_or_proposed_section is not None \
                and type(boundaries_or_proposed_section) is not dict:
            boundaries = boundaries_or_proposed_section
        elif boundaries is None:
            if proposed_section is not None:
                boundaries = proposed_section['box']
            else:
                boundaries = [0, 0, 0, 0]
        self.proposed_section = proposed_section
        self.diagonal_points = boundaries
        self.section_container = container
        if threshold is None:
            if self.section_container is None:
                threshold = 40
            else:
                threshold = self.section_container._threshold
        if type(threshold) is ThresholdAttribute:
            self._threshold = threshold
        else:
            self._threshold = ThresholdAttribute()
            if type(threshold) is int:
                self.threshold = threshold
        self._is_bottom_expandable = is_bottom_expandable

    @property
    def threshold(self):
        return self._threshold.threshold

    @threshold.setter
    def threshold(self, v):
        self._threshold.threshold = v

    @property
    def coordinates(self):
        return [self.left, self.top, self.right, self.bottom]

    @property
    def diagonal_points(self):
        return self._diagonal_points

    @property
    def lt_coord(self):
        return self.diagonal_points[0]

    @property
    def rb_coord(self):
        return self.diagonal_points[1]

    @property
    def left(self):
        return self.diagonal_points[0][0]

    @property
    def top(self):
        return self.diagonal_points[0][1]

    @property
    def right(self):
        return self.diagonal_points[1][0]

    @property
    def bottom(self):
        return self.diagonal_points[1][1]

    @diagonal_points.setter
    def diagonal_points(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
        x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
        self._diagonal_points = [[x1, y1], [x2, y2]]

    @lt_coord.setter
    def lt_coord(self, p1: Union[int, list] = 0, y=0):
        if type(p1) is list:
            x = p1[0]
            y = p1[1]
        elif type(p1) is int:
            x = p1

        self._diagonal_points[0] = [x, y]

    @rb_coord.setter
    def rb_coord(self, p1: Union[int, list] = 100000, y=100000):
        if type(p1) is list:
            x = p1[0]
            y = p1[1]
        elif type(p1) is int:
            x = p1

        self._diagonal_points[1] = [x, y]

    @left.setter
    def left(self, x=0):
        self._diagonal_points[0][0] = x

    @top.setter
    def top(self, y=0):
        self._diagonal_points[0][1] = y

    @right.setter
    def right(self, x=100000):
        self._diagonal_points[1][0] = x

    @bottom.setter
    def bottom(self, y=100000):
        self._diagonal_points[1][1] = y

    @property
    def width(self):
        return self.right - self.left

    def is_empty(self):
        dif = self.bottom - self.top
        return dif < self.threshold

    @property
    def is_bottom_expandable(self):
        result = self._is_bottom_expandable
        if result is None:
            result = self.section_container.is_bottom_expandable
        return result

    @is_bottom_expandable.setter
    def is_bottom_expandable(self, val):
        self._is_bottom_expandable = val

    def can_be_added(self, writing_area_properties):
        pass

    def add_writing_area(self, writing_area_properties):
        pass


class StructuredSection(AbstractSection):
    """
    A StructuredSection class implements a structured section which can contain 2 kinds of structures: section stack or
    sibling sections

    Attributtes
    -----------


    Methods
    -------

    """

    def __init__(self, boundaries_or_proposed_section: Union[dict, list] = None,
                 threshold: Union[int, ThresholdAttribute] = None, proposed_section=None, boundaries=None,
                 container=None):
        super().__init__(boundaries_or_proposed_section, threshold, container, proposed_section, boundaries,
                         is_bottom_expandable=True)
        self._sections = []
        self.is_right_expandable = False
        self.is_bottom_expandable = True

    @property
    def sections(self):
        return self._sections

    def get_single_sections_as_boxes(self, boxes=[]):
        for section in self.sections:
            section.get_single_sections_as_boxes(boxes)
        return boxes

    def add_new_section(self, section=False):
        self._sections.append(section)

    # def is_area_inside(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
    #     x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
    #     edges = [0, 1]
    #     if not self.is_rigth_expandable:
    #         edges.append(2)
    #     if not self.is_bottom_expandable:
    #         edges.append(3)
    #     return contains(edges, self.coordinates, (x1, y1, x2, y2), self.threshold)


# def _compare(item1, item2):
#     if item1["is_single"] or item2["is_single"]:
#         ret = item1["area"][1] - item2["area"][1]
#     elif _areas_belong_to_the_same_columns(item1["area"], item2["area"], item1, item2, 60):
#         ret = item1["area"][1] - item2["area"][1]
#     else:
#         ret = item1["guess_left"] - item2["guess_left"]
#     return ret


class MainLayout(StructuredSection):
    def __init__(self, w: int, h: int = 0, threshold=40, proposed_sections=None):
        super().__init__([0, 0, w, h], threshold)
        self.proposed_sections = proposed_sections

    def get_single_sections_as_boxes(self):
        boxes = []
        return super().get_single_sections_as_boxes(boxes)

    def search_section(self, writing_area_properties):
        pos = -1
        for i, section in enumerate(self.sections):
            if section.can_be_added(writing_area_properties):
                pos = i
                break
        return pos

    def _has_area_similar_width_to_proposed_columns(self, writing_area_properties, proposed_section):
        if len(proposed_section['columns']) > 0:
            margin = self.threshold + self.threshold + self.threshold
            width_sibling = proposed_section['columns'][0][2] - proposed_section['columns'][0][0]
            if width_sibling == -1:
                dif = 0
            else:
                x1, y1, x2, y2 = writing_area_properties['area']
                dif = abs(x2 - x1 - width_sibling)
                if dif > margin:
                    dif = abs(writing_area_properties['guess_width'] - width_sibling)
            ret = dif <= margin
        else:
            ret = False
        return ret

    def _has_area_similar_center_to_proposed_columns(self, writing_area_properties, proposed_section):
        if len(proposed_section['columns']) > 0:
            margin = self.threshold + self.threshold
            x1, y1, x2, y2 = writing_area_properties['area']
            width_sibling = proposed_section['columns'][0][2] - proposed_section['columns'][0][2]
            dif = abs(width_sibling / 2 - (x1 + x2) / 2) % width_sibling
            ret = dif <= margin
        else:
            ret = False
        return ret

    def is_proposed_section_compatible(self, writing_area_properties, proposed_section):
        result = writing_area_properties['is_single'] == (len(proposed_section['columns']) == 0)
        result = result and contains([0, 1, 2, 3], proposed_section['box'], writing_area_properties['area'],
                                     self.threshold)
        if not writing_area_properties['is_single']:
            sw = self._has_area_similar_width_to_proposed_columns(writing_area_properties, proposed_section)
            if not sw:
                sw = self._has_area_similar_center_to_proposed_columns(writing_area_properties, proposed_section)
            result = result and sw
        return result

    def search_proposed_section(self, writing_area_properties):
        pos = -1
        for i, proposed_section in enumerate(self.proposed_sections):
            if self.is_proposed_section_compatible(writing_area_properties, proposed_section):
                pos = i
                break
        return pos

    def add_section_from_proposed_section(self, pos_proposed_section):
        proposed_section = self.proposed_sections[pos_proposed_section]
        if len(proposed_section['columns']) == 0:
            section = SingleSection(self, proposed_section=proposed_section, is_bottom_expandable=True)
        else:
            section = BigSectionOfSibling(self, proposed_section=proposed_section)
        for prev_section in self.sections:
            if prev_section.is_bottom_expandable and horizontal_overlapping_ratio(section.coordinates,
                                                                                  prev_section.coordinates,
                                                                                  self.threshold) >= 0.9:
                prev_section.is_bottom_expandable = False
        self.add_new_section(section)
        self.proposed_sections.pop(pos_proposed_section)

    def add_writing_area(self, writing_area_properties):
        pos = self.search_section(writing_area_properties)
        if pos >= 0:
            self.sections[pos].add_writing_area(writing_area_properties)
        else:
            ppos = self.search_proposed_section(writing_area_properties)
            if ppos >= 0:
                pos = len(self.sections)
                self.add_section_from_proposed_section(ppos)
                self.sections[pos].top = writing_area_properties['area'][1]
                self.sections[pos].add_writing_area(writing_area_properties)
            # elif writing_area_properties['is_single']:
            # create nes extra section

    # def adjust_size_of_sections(self):
    #     #adjust vertically
    #     #order bigsection vertically
    #     sorted(self.sections, key=lambda x: x.top*10000+x.left)
    #     for pos in range(len(self.sections)):
    #         offset = 1
    #         while pos+offset<len(self.sections) and overlap_vertically(self.sections[pos].coodinates,
    #                                                                    self.sections[pos+offset].coodinates):
    #             offset += 1
    #         if offset<len(self.sections):
    #             #need adjust
    #             self.sections[pos].bottom = (self.sections[pos].bottom + self.sections[pos+offset].top)//2
    #

    @staticmethod
    def build_lauoud_from_sections(sections, writing_area_list, w: int, h: int, threshold=None):
        main_layout = MainLayout(w, h, proposed_sections=sections)
        if threshold is not None:
            main_layout.threshold = threshold

        min_left = 100000
        max_right = 0
        for writing_area in writing_area_list:
            if min_left > writing_area[0]:
                min_left = writing_area[0]
            if max_right < writing_area[2]:
                max_right = writing_area[2]

        writing_area_list.sort(key=lambda x: x[1] * 10000 + x[0])
        for i, writing_area in enumerate(writing_area_list):
            guess_left = min_left
            guess_right = max_right
            is_single = True
            offset = -1
            # while (i + offset) >= 0 and overlap_vertically(writing_area_list[i + offset],
            #                                                writing_area, main_layout.threshold):
            while (i + offset) >= 0:
                if overlap_vertically(writing_area_list[i + offset], writing_area, main_layout.threshold):
                    if writing_area_list[i + offset][0] < writing_area[0] and guess_left < writing_area_list[i + offset][2]:
                        guess_left = writing_area_list[i + offset][2]
                    elif writing_area[0] <= writing_area_list[i + offset][0] < guess_right:
                        guess_right = writing_area_list[i + offset][0]
                    is_single = False
                offset -= 1
            offset = 1
            # while (i + offset) < len(writing_area_list) and overlap_vertically(writing_area_list[i + offset],
            #                                                                    writing_area, main_layout.threshold):
            while (i + offset) < len(writing_area_list):
                if overlap_vertically(writing_area_list[i + offset], writing_area, main_layout.threshold):
                    if writing_area_list[i + offset][0] < writing_area[0] and guess_left < writing_area_list[i + offset][2]:
                        guess_left = writing_area_list[i + offset][2]
                    elif writing_area[0] <= writing_area_list[i + offset][0] < guess_right:
                        guess_right = writing_area_list[i + offset][0]
                    is_single = False
                offset += 1
            guess_width = guess_right - guess_left
            wa = {"area": writing_area, "guess_left": guess_left, "guess_right": guess_right,
                  "guess_width": guess_right - guess_left, "is_single": is_single}
            main_layout.add_writing_area(wa)

        for i, section in enumerate(main_layout.sections):
            if i == 0:
                section.top = main_layout.top
            elif i == len(main_layout.sections) - 1:
                section.bottom = main_layout.bottom
            if i < len(main_layout.sections) - 1:
                section.bottom = main_layout.sections[i + 1].top
            section.left = main_layout.left
            section.right = main_layout.right
            if type(section) is BigSectionOfSibling:
                for j, col in enumerate(section.siblings):
                    if j == 0:
                        col.left = main_layout.left
                    if j == len(section.siblings) - 1:
                        col.right = main_layout.right
                    if j < len(section.siblings) - 1:
                        col.right = section.siblings[j + 1].left = (col.right + section.siblings[j + 1].left) // 2

        return main_layout

    @staticmethod
    def build_layout(writing_area_list, w: int, threshold=None):
        main_layout = MainLayout(w)
        if threshold is not None:
            main_layout.threshold = threshold

        min_left = 100000
        max_right = 0
        for writing_area in writing_area_list:
            if min_left > writing_area[0]:
                min_left = writing_area[0]
            if max_right < writing_area[2]:
                max_right = writing_area[2]

        writing_area_list.sort(key=lambda x: x[1] * 10000 + x[0])
        # for i, writing_area in enumerate(writing_area_list):
        #     guess_left = min_left
        #     guess_right = max_right
        #     is_single = True
        #     offset = -1
        #     while (i + offset) >= 0 and overlap_vertically(writing_area_list[i + offset],
        #                                                    writing_area, main_layout.threshold):
        #         if writing_area_list[i + offset][0] < writing_area[0] and guess_left < writing_area_list[i + offset][2]:
        #             guess_left = writing_area_list[i + offset][2]
        #         elif writing_area[0] <= writing_area_list[i + offset][0] < guess_right:
        #             guess_right = writing_area_list[i + offset][0]
        #         is_single = False
        #         offset -= 1
        #     offset = 1
        #     while (i + offset) < len(writing_area_list) and overlap_vertically(writing_area_list[i + offset],
        #                                                                        writing_area, main_layout.threshold):
        #         if writing_area_list[i + offset][0] < writing_area[0] and guess_left < writing_area_list[i + offset][2]:
        #             guess_left = writing_area_list[i + offset][2]
        #         elif writing_area[0] <= writing_area_list[i + offset][0] < guess_right:
        #             guess_right = writing_area_list[i + offset][0]
        #         is_single = False
        #         offset += 1
        #     writing_area_properties_list.append(
        #         {"area": writing_area_list[i], "guess_left": guess_left, "guess_right": guess_right,
        #          "guess_width": guess_right - guess_left, "is_single": is_single})
        #
        # writing_area_list.sort(key=lambda x: x[1] * 10000 + x[0])
        #
        # # writing_area_list = [x["area"] for x in sorted(writing_area_properties_list, key=cmp_to_key(_compare))]

        print(f"ordered:{writing_area_list}")

        lef_unexplored = min_left
        top_unexplored = 0
        right_unexplored = max_right
        for i, writing_area in enumerate(writing_area_list):
            added = False
            guess_left = min_left
            guess_right = max_right
            # force_sibling = False
            is_single = True
            offset = -1
            while (i + offset) >= 0 and overlap_vertically(writing_area_list[i + offset],
                                                           writing_area, main_layout.threshold):
                if writing_area_list[i + offset][0] < writing_area[0] and guess_left < writing_area_list[i + offset][2]:
                    guess_left = writing_area_list[i + offset][2]
                elif writing_area[0] <= writing_area_list[i + offset][0] < guess_right:
                    guess_right = writing_area_list[i + offset][0]
                # force_sibling = True
                is_single = False
                offset -= 1
            offset = 1
            while (i + offset) < len(writing_area_list) and overlap_vertically(writing_area_list[i + offset],
                                                                               writing_area, main_layout.threshold):
                if writing_area_list[i + offset][0] < writing_area[0] and guess_left < writing_area_list[i + offset][2]:
                    guess_left = writing_area_list[i + offset][2]
                elif writing_area[0] <= writing_area_list[i + offset][0] < guess_right:
                    guess_right = writing_area_list[i + offset][0]
                is_single = False
                offset += 1
            guess_width = guess_right - guess_left
            pos = len(main_layout.sections) - 1
            # if force_sibling:
            #     added, lef_unexplored, top_unexplored, right_unexplored = main_layout.sections[pos].try_to_add_writing_area(
            #         writing_area, guess_width=guess_width, force_sibling=force_sibling)
            # elif is_single:
            if is_single:
                # need single section
                if pos > -1 and type(main_layout.sections[pos]) is SingleSection:
                    # OK
                    main_layout.sections[pos].try_to_add_writing_area(writing_area, guess_width=guess_width)
                    section_pos = pos
                else:
                    if pos < 0:
                        t = 0
                    else:
                        t = main_layout.sections[pos].bottom

                    # create new SingleSection
                    new_section = SingleSection(main_layout, min_left, t, max_right, t, guess_width,
                                                threshold=main_layout._threshold)
                    new_section.try_to_add_writing_area(writing_area, guess_width=guess_width)
                    main_layout.add_new_section(new_section)
                    section_pos = len(main_layout.sections) - 1
            else:
                found = False
                while not found and pos >= 0:
                    if type(main_layout.sections[pos]) is SingleSection:
                        pos = -1
                        break
                    found = type(main_layout.sections[pos]) is BigSectionOfSibling \
                            and main_layout.sections[pos].is_area_inside(writing_area)
                    # if overlap_horizontally(
                    #         [main_layout.sections[pos].left, 0,  main_layout.sections[pos].right, 0],
                    #         writing_area, main_layout.threshold):
                    #     pos = 0
                    pos -= 1
                if found:
                    pos += 1
                    added, lef_unexplored, top_unexplored, right_unexplored = main_layout.sections[
                        pos].try_to_add_writing_area(
                        writing_area, guess_width=guess_width)
                if added:
                    section_pos = pos
                else:
                    if lef_unexplored + main_layout.threshold > writing_area[0]:
                        lef_unexplored = min_left
                    if right_unexplored - main_layout.threshold < writing_area[2]:
                        right_unexplored = max_right
                    new_section = BigSectionOfSibling(main_layout, lef_unexplored, writing_area[1], right_unexplored,
                                                      top_unexplored,
                                                      threshold=main_layout._threshold)
                    new_section.try_to_add_writing_area(writing_area, guess_width=guess_width)
                    main_layout.add_new_section(new_section)
                    section_pos = len(main_layout.sections) - 1

            # Adapatar l'amplada de les seccions veines a la mida de l'actual si se sobreposen verticalment
            # for p in range(len(main_layout.sections)):
            #     if p != section_pos:
            #         if overlap_vertically(main_layout.sections[p].coordinates, main_layout.sections[section_pos].coordinates, main_layout.threshold):

        # main_layout.build_layout_structure(writing_area_list, 0)
        return main_layout


class BigSectionOfSibling(StructuredSection):
    def __init__(self, container: MainLayout, boundaries_or_proposed_section: Union[dict, list] = None,
                 threshold: Union[int, ThresholdAttribute] = None, proposed_section=None, boundaries=None):
        super().__init__(boundaries_or_proposed_section, threshold, proposed_section, boundaries, container)
        if self.proposed_section is not None and len(self.proposed_section['columns']) > 0:
            self._width_sibling = self.proposed_section['columns'][0][2] - self.proposed_section['columns'][0][0]
            for col in self.proposed_section['columns']:
                col_sec = SingleSection(self, col)
                self.add_new_section(col_sec)
        else:
            self._width_sibling = -1

    @AbstractSection.top.setter
    def top(self, value):
        self._diagonal_points[0][1] = value
        for sib in self.siblings:
            sib.top = value

    @property
    def width_sibling(self):
        return self._width_sibling

    @property
    def siblings(self):
        return self.sections

    def _has_area_similar_width(self, writing_area_properties):
        margin = self.threshold + self.threshold + self.threshold
        if self.width_sibling == -1:
            dif = 0
        else:
            x1, y1, x2, y2 = writing_area_properties['area']
            dif = abs(x2 - x1 - self.width_sibling)
            if dif > margin:
                dif = abs(writing_area_properties['guess_width'] - self.width_sibling)
        return dif <= margin

    def _has_area_similar_center(self, writing_area_properties):
        margin = self.threshold + self.threshold
        x1, y1, x2, y2 = writing_area_properties['area']
        dif = abs(self.width_sibling / 2 - (x1 + x2) / 2) % self.width_sibling
        return dif <= margin

    def is_area_compatible(self, writing_area_properties):
        ret = self._has_area_similar_width(writing_area_properties)
        if not ret:
            ret = self._has_area_similar_center(writing_area_properties)
        return ret

    def search_sibling(self, writing_area_properties):
        pos = -1
        for i, sibling in enumerate(self.siblings):
            if sibling.can_be_added(writing_area_properties):
                pos = i
                break
        return pos

    def can_be_added(self, writing_area_properties):
        result = not writing_area_properties['is_single']
        edges = [0, 1, 2]
        if not self.is_bottom_expandable:
            edges.append(3)
        result = result and contains(edges, self.coordinates, writing_area_properties['area'], self.threshold)
        result = result and self.is_area_compatible(writing_area_properties)
        pos = self.search_sibling(writing_area_properties)
        return result and pos >= 0

    def add_writing_area(self, writing_area_properties):
        pos = self.search_sibling(writing_area_properties)
        if pos == -1:
            raise "Error trying add writing area to wrong column"
        self.siblings[pos].add_writing_area(writing_area_properties)


class SingleSection(AbstractSection):
    """
    Class used to represent a single section. In real world, a single section has not layout structure and can only
    contain text.
    This implementation maintains only the rectangle information.

    Attributtes
    -----------


    Methods
    -------

    """

    def __init__(self, container: StructuredSection, boundaries_or_proposed_section: Union[dict, list] = None,
                 threshold: Union[int, ThresholdAttribute] = None, proposed_section=None, boundaries=None,
                 is_bottom_expandable=None):
        super().__init__(boundaries_or_proposed_section, threshold=threshold, container=container,
                         proposed_section=proposed_section, boundaries=boundaries,
                         is_bottom_expandable=is_bottom_expandable)
        self._len = 0
        self._suma_center = 0
        self._is_column = type(container) is BigSectionOfSibling

    @property
    def center(self):
        return self._suma_center / self._len

    def can_be_added(self, writing_area_properties):
        result = writing_area_properties['is_single'] == (not self._is_column)
        edges = [0, 1, 2]
        if not self.is_bottom_expandable:
            edges.append(3)
        result = result and contains(edges, self.coordinates, writing_area_properties['area'], self.threshold)
        return result

    def add_writing_area(self, writing_area_properties):
        x1, y1, x2, y2 = writing_area_properties['area']
        self.bottom = y2
        self._len += 1
        self._suma_center += (x1 + x2) / 2
        if x1 < self.left:
            self.left = x1
        if x2 > self.right:
            self.right = x2

    def get_single_sections_as_boxes(self, boxes=[]):
        boxes.append(self.coordinates)

    def get_compatible_status(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
        ret = super().get_compatible_status(p1, p2, x2, y2)
        if ret != 0:
            x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
            ret = self.get_compatible_status_as_guess(x1, y1, x2, y2)
            if ret != 0:
                c = (x1 + x2) / 2
                ldif = c - self.center
                if abs(ldif) <= self.threshold:
                    ret = 0
                elif ldif < self.threshold:
                    ret = -1
                else:
                    ret = 1
        return ret

    def get_compatible_status_as_guess(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
        ret = 0
        x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
        ldif = x1 - self.guess_left + self.threshold
        rdif = self.guess_right - x2 + self.threshold
        if ldif >= 0 and rdif >= 0:
            ret = 0
        elif ldif < 0:
            ret = -1
        else:
            ret = 1
        return ret


def test_build_layout():
    m = MainLayout.build_layout([[10, 10, 90, 50], [10, 50, 100, 60], [10, 60, 90, 70], [10, 72, 90, 85],
                                 [10, 85, 90, 90], [10, 91, 45, 98], [55, 90, 90, 105], [10, 110, 45, 120],
                                 [55, 104, 90, 117], [10, 119, 90, 125], [10, 125, 90, 135]], 105, 2)
    boxes = m.get_single_sections_as_boxes()
    print(boxes)


def test():
    # single structure

    main_layout = MainLayout(100, 2)
    main_layout.add_new_section(SingleSection())
    main_layout.try_to_add_writing_area(10, 10, 90, 50)
    main_layout.try_to_add_writing_area(10, 50, 90, 60)
    try:
        main_layout.try_to_add_writing_area(10, 50, 90, 60)
    except ValueError as e:
        print(e)

    main_layout.try_to_add_writing_area(10, 60, 90, 70)
    main_layout.add_new_section(SingleSection())
    main_layout.try_to_add_writing_area(10, 72, 90, 85)
    main_layout.try_to_add_writing_area(10, 85, 90, 90)
    main_layout.add_new_section(BigSectionOfSibling())
    main_layout.try_to_add_writing_area(10, 91, 40, 98)
    main_layout.try_to_add_writing_area(55, 94, 90, 105)
    main_layout.add_new_section(SingleSection())
    main_layout.try_to_add_writing_area(10, 104, 90, 120)

    print(main_layout.diagonal_points)

    boxes = main_layout.get_single_sections_as_boxes()

    print(boxes)

    m = MainLayout.build_layout([[10, 10, 90, 50], [10, 50, 100, 60], [10, 60, 90, 70], [10, 72, 90, 85],
                                 [10, 85, 90, 90], [10, 91, 40, 98], [55, 94, 90, 105], [10, 110, 40, 120],
                                 [55, 104, 90, 117], [10, 119, 90, 125], [10, 125, 90, 135]], 105)
    boxes = m.get_single_sections_as_boxes()
    print(boxes)


if __name__ == "__main__":
    test()
    test_build_layout()
