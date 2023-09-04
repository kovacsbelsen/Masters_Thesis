import os
import networkx as nx
import pygame
import pygame.font
import random
from collections import namedtuple
import math
from collections import deque
import matplotlib.pyplot as plt
import json
import time
from concurrent.futures import ProcessPoolExecutor
import re
from multiprocessing import Process
from multiprocessing import Pool
import pandas as pd
import agentpy as ap

# 80 square meters, 10m x 8m, 1000 x 800, 1m = 100
RoomType = namedtuple('RoomType', ['name', 'min_size', 'max_size', 'color', 'privacy'])
ROOM_TYPES = [
    RoomType('bedroom', (200, 200), (200, 500), (0, 0, 255),'private'),
    RoomType('living_room', (300, 400), (300, 700), (0, 255, 0),'common'),
    RoomType('kitchen', (200, 350), (300, 500), (255, 0, 0),'common'),
    RoomType('bathroom', (150, 200), (150, 250), (255, 155, 20),'private')
]

def room_type_to_dict(room_type):
    return {
        "name": room_type.name,
        "min_size": room_type.min_size,
        "max_size": room_type.max_size,
        "color": room_type.color,
        "privacy": room_type.privacy
    }

def room_type_from_dict(data):
    return RoomType(
        name=data['name'],
        min_size=tuple(data["min_size"]),
        max_size=tuple(data["max_size"]),
        color=tuple(data["color"]),
        privacy = data['privacy']
    )    


class ObjectType:
    def __init__(self, name, effect, low=1, high=10):
        self.name = name
        self.effect = effect
        self.interaction_time = round(random.uniform(low, high), 1)

    def to_dict(self):
        return {
            "name": self.name,
            "effect": self.effect
        }
    
    @classmethod
    def from_dict(cls, data):
        name = data['name']
        effect = data['effect']
        return cls(name, effect)

BED = ObjectType("Bed", {"energy": 50}, low=4, high=8)
TV = ObjectType("TV", {"fun": 40}, low=0.5, high=4)
COMPUTER = ObjectType("COMPUTER", {"fun": 40}, low=0.5, high=4)
TOILET = ObjectType("Toilet", {"bladder": 60}, low=0.1, high=0.5)
FRIDGE = ObjectType("Fridge", {"hunger": 50}, low=0.2, high=1)
SINK = ObjectType("Sink", {"thirst": 50}, low=0.1, high=0.1)
SHOWER = ObjectType("Shower", {"hygiene": 60}, low=0.1, high=0.5)

OBJECT_TYPES = {
    "bedroom": [BED],
    "living_room": [TV, COMPUTER],
    "bathroom": [TOILET, SHOWER],
    "kitchen": [FRIDGE, SINK]
}


class Object:
    def __init__(self, room, object_type):
        self.name = object_type.name
        self.room = room
        self.object_type = object_type
        self.interaction_time = object_type.interaction_time
        self.is_interacted = False
        self.interacting_agent = None
        self.queue = []

        # Grid positions inside the room
        grid_positions = [
            (room.rect.x + room.rect.width * 0.25, room.rect.y + room.rect.height * 0.25),
            (room.rect.x + room.rect.width * 0.75, room.rect.y + room.rect.height * 0.25),
            (room.rect.x + room.rect.width * 0.25, room.rect.y + room.rect.height * 0.5),
            (room.rect.x + room.rect.width * 0.75, room.rect.y + room.rect.height * 0.5),
            (room.rect.x + room.rect.width * 0.25, room.rect.y + room.rect.height * 0.75),
            (room.rect.x + room.rect.width * 0.75, room.rect.y + room.rect.height * 0.75),
        ]

        # Choose a random position for the object
        self.position = random.choice(grid_positions)

    def is_free(self):
        return not self.is_interacted

    def to_dict(self):
        return {
            "name": self.name,
            "object_type": self.object_type.to_dict(),
            "interaction_time": self.interaction_time,
            "position": self.position,
        }
    
    @classmethod
    def from_dict(cls, data, room):
        object_type_data = data['object_type']
        object_type = ObjectType.from_dict(object_type_data)
        object_instance = cls(room, object_type)
        object_instance.name = data['name']
        object_instance.interaction_time = data['interaction_time']
        object_instance.position = tuple(data['position'])
        return object_instance


    def draw(self, surface):
        # Draw the object at its position
        pygame.draw.circle(surface, (50, 155, 150), self.position, 10)

        inf = object_font.render(self.name + " " + str(self.position), True, (0, 0, 0))
        screen.blit(inf, self.position)

class Window(pygame.Rect):
    def __init__(self, left, top, length, thickness):
        super().__init__(left, top, length, thickness)
        self.color = (0, 150, 255)

def create_windows(screen):
    screen_width, screen_height = screen.get_size()
    window_length = 100
    window_thickness = 20

    # Create windows on the top edge
    top_windows = [
        Window(100, 0, window_length, window_thickness),
        Window(screen_width // 2 - window_length // 2, 0, window_length, window_thickness),
        Window(screen_width - 100 - window_length, 0, window_length, window_thickness)
    ]

    # Create windows on the bottom edge
    bottom_windows = [
        Window(100, screen_height - window_thickness, window_length, window_thickness),
        Window(screen_width // 2 - window_length // 2, screen_height - window_thickness, window_length, window_thickness),
        Window(screen_width - 100 - window_length, screen_height - window_thickness, window_length, window_thickness)
    ]

    return top_windows + bottom_windows

class Door:
    def __init__(self, position, orientation, room1, room2):
        self.position = position
        self.orientation = orientation
        self.room1 = room1
        self.room2 = room2

    def to_dict(self):
        return {
            "position": self.position,
            "orientation": self.orientation,
            "room1": self.room1.to_dict(),
            "room2": self.room2.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        position = tuple(data["position"])
        orientation = data["orientation"]
        room1 = SlicingNode.from_dict(data["room1"])
        room2 = SlicingNode.from_dict(data["room2"])
        return cls(position, orientation, room1, room2)

class Room:
    def __init__(self, node, rect):
        self.node = node
        self.rect = rect
        self.neighbors = []
        self.objects = []  # Add the objects attribute
        self.add_objects()  # Add objects to the room
        self.doors = []
        self.window = False
        self.size = None

    def add_objects(self):
        object_types = OBJECT_TYPES.get(self.node.room_type.name)
        if object_types:
            for object_type in object_types:
                self.objects.append(Object(self, object_type))

    @classmethod
    def from_dict(cls, data):
        node = SlicingNode.from_dict(data['node'])  # Create a SlicingNode instance
        rect = pygame.Rect(*data['rect'])  # Create a pygame.Rect object from the list of values
        room_instance = cls(node, rect)
        room_instance.objects = [Object.from_dict(obj_data, room_instance) for obj_data in data['objects']]
        room_instance.doors = [Door.from_dict(door_data) for door_data in data['doors']]
        return room_instance


    # Add the to_dict method
    def to_dict(self):
        return {
            "node": self.node.to_dict(),
            "rect": [self.rect.x, self.rect.y, self.rect.width, self.rect.height],  # Store rect as a list of values
            "neighbors": [neighbor.node.to_dict() for neighbor in self.neighbors],
            "objects": [obj.to_dict() for obj in self.objects],
            "doors": [door.to_dict() for door in self.doors],
        }


class SlicingNode:
    def __init__(self, value, cut_percentage, room_type):
        self.value = value
        self.children = []
        self.cut_percentage = cut_percentage
        self.room_type = room_type

    def to_dict(self):
        return {
            "value": self.value,
            "cut_percentage": self.cut_percentage,
            "children": [child.to_dict() for child in self.children],
            "room_type": room_type_to_dict(self.room_type) if self.room_type else None
        }

    @classmethod
    def from_dict(cls, data):
        value = data['value']
        cut_percentage = data['cut_percentage']
        room_type = room_type_from_dict(data["room_type"]) if data["room_type"] is not None else None
        node_instance = cls(value, cut_percentage, room_type)
        node_instance.children = [cls.from_dict(child_data) for child_data in data['children']]
        return node_instance                

def identify_neighbors(rooms):
    for i, room1 in enumerate(rooms):
        for j, room2 in enumerate(rooms):
            if i != j and are_neighbors(room1.rect, room2.rect):
                room1.neighbors.append(room2)


def are_neighbors(rect1, rect2): 
    return rect1.colliderect(rect2.inflate(10, 10))

def check_door_between_rooms(r1,r2):
    for door in r1.doors:
        if door.room1.value == r2.node.value:
            return True
        if door.room2.value == r2.node.value:
            return True
    for door in r2.doors:
        if door.room1.value == r1.node.value:
            return True
        if door.room2.value == r1.node.value:
            return True
    return False

def get_door_between_rooms(r1,r2):
    for door in r1.doors:
        if door.room1.value == r2.node.value:
            return door
        if door.room2.value == r2.node.value:
            return door
    for door in r2.doors:
        if door.room1.value == r1.node.value:
            return door
        if door.room2.value == r1.node.value:
            return door
    return None

        
class GraphVisualization:
    def __init__(self):
        self.visual = []

    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)

        # Create a dictionary with room objects as keys and their room types as values
        labels = {room: f"{room.node.room_type.name} {room.node.value}" for room in G.nodes()}

        # Draw the graph with the labels
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos, with_labels = False)
        nx.draw_networkx_labels(G, pos, labels)

        plt.show()

    @classmethod
    def from_floor_plan_graph(cls, floor_plan_graph):
        graph_visualization = cls()

        for room, neighbors in floor_plan_graph.items():
            for neighbor in neighbors:
                graph_visualization.addEdge(room, neighbor)

        return graph_visualization




def add_doors(room, screen, doors_list, graph):
    for neighbor in room.neighbors:
        if not check_door_between_rooms(room, neighbor):

            rect1, rect2 = room.rect, neighbor.rect
            #if rect1.y == rect2.y:  # vertical shared edge
            if abs(rect1.x - (rect2.x + rect2.width)) < 5  or abs(rect2.x - (rect1.x + rect1.width)) < 5:
                door_x = max(rect1.x, rect2.x)
                door_y = (max(rect1.y, rect2.y) + min(rect1.y + rect1.height, rect2.y + rect2.height)) * 0.5
                door_position = (door_x, door_y)
                door = Door(door_position, 'vertical', room.node, neighbor.node)
                # check door position to solve duplicate
                if door not in room.doors and door not in neighbor.doors:
                    room.doors.append(door)
                    neighbor.doors.append(door)
                    draw_door(door, screen)
                    doors_list.append(door)
                    # If rooms are neighbors, add an edge in the graph (undirected)
                    graph[room].append(neighbor)
                    graph[neighbor].append(room)
                    return doors_list, graph

            #elif rect1.x == rect2.x:  # horizontal shared edge
            elif abs(rect1.y - (rect2.y + rect2.height))< 4 or abs(rect2.y -(rect1.y + rect1.height)) < 4:
                door_x = (max(rect1.x, rect2.x) + min(rect1.x + rect1.width, rect2.x+ rect2.width)) * 0.5
                door_y = max(rect1.y, rect2.y)
                door_position = (door_x, door_y)
                door = Door(door_position, 'horizontal', room.node, neighbor.node)
                
                if door not in room.doors and door not in neighbor.doors:
                    room.doors.append(door)
                    neighbor.doors.append(door)
                    draw_door(door, screen)
                    doors_list.append(door)
                    # If rooms are neighbors, add an edge in the graph (undirected)
                    graph[room].append(neighbor)
                    graph[neighbor].append(room)
                    return doors_list, graph

def draw_door(door, screen):
    door_length = 30
    door_thickness = 10

    if door.orientation == 'horizontal':
        door_rect = pygame.Rect(door.position[0] - door_length // 2, door.position[1] - door_thickness // 2, door_length, door_thickness)
    elif door.orientation == 'vertical':
        door_rect = pygame.Rect(door.position[0] - door_thickness // 2, door.position[1] - door_length // 2, door_thickness, door_length)

    pygame.draw.rect(screen, (128, 128, 128), door_rect)
    inf = object_font.render(door.room1.room_type.name + " " + str(door.room1.value) + " " +  door.room2.room_type.name + " " + str(door.room2.value), True, (0, 0, 0))
    screen.blit(inf, (door.position[0], door.position[1]))


def generate_slicing_tree(num_rooms, room_types):
    if num_rooms == 1:
        node = SlicingNode(1, random.uniform(0.2, 0.8), random.choice(room_types))
        return node

    rooms = list(range(1, num_rooms + 1))
    random.shuffle(rooms)

    return generate_slicing_tree_from_rooms(rooms, room_types)

def generate_slicing_tree_from_rooms(rooms, room_types):
    if len(rooms) == 1:
        node = SlicingNode(rooms[0], random.uniform(0.35, 0.65), random.choice(room_types))
        return node

    split_type = random.choice(['H', 'V'])

    left_rooms_count = random.randint(1, len(rooms) - 1)
    right_rooms_count = len(rooms) - left_rooms_count

    left_rooms = set()
    right_rooms = set()

    available_rooms = set(rooms)

    while len(left_rooms) < left_rooms_count:
        room = random.choice(list(available_rooms))
        left_rooms.add(room)
        available_rooms.remove(room)

    right_rooms = available_rooms

    left_tree = generate_slicing_tree_from_rooms(list(left_rooms), room_types)
    right_tree = generate_slicing_tree_from_rooms(list(right_rooms), room_types)

    node = SlicingNode(split_type, random.uniform(0.2, 0.8), None)
    node.children = [left_tree, right_tree]
    return node

def create_floorplan(node, rect, rooms=None):
    if rooms is None:
        rooms = []

    if not node:
        return rooms

    if not hasattr(node, 'children') or not node.children:  # Leaf node
        room = Room(node, rect)
        if room not in rooms:
            rooms.append(room)
        return rooms

    left, right = node.children

    if node.value == 'V':  # Vertical cut
        cut_x = rect.x + rect.width * node.cut_percentage
        left_rect = pygame.Rect(rect.x, rect.y, rect.width * node.cut_percentage, rect.height)
        right_rect = pygame.Rect(cut_x, rect.y, rect.width * (1 - node.cut_percentage), rect.height)

        # Add door
        door_position = (cut_x, rect.y + rect.height * 0.5)
        node.door = Door(door_position, 'vertical', left, right)

    elif node.value == 'H':  # Horizontal cut
        cut_y = rect.y + rect.height * node.cut_percentage
        left_rect = pygame.Rect(rect.x, rect.y, rect.width, rect.height * node.cut_percentage)
        right_rect = pygame.Rect(rect.x, cut_y, rect.width, rect.height * (1 - node.cut_percentage))

        # Add door
        door_position = (rect.x + rect.width * 0.5, cut_y)
        node.door = Door(door_position, 'horizontal', left, right)

    rooms_left = create_floorplan(left, left_rect, rooms)
    rooms_right = create_floorplan(right, right_rect, rooms)

    return rooms_left + [r for r in rooms_right if r not in rooms_left]

def draw_floorplan(rooms, screen):
    for room in rooms:
        rect = room.rect
        node = room.node

        pygame.draw.rect(screen, node.room_type.color, rect, 0)  # Fill the rectangle
        pygame.draw.rect(screen, (0, 0, 0), rect, 1)  # Draw the outline

        # Draw room name and number
        font = pygame.font.Font(None, 24)
        room_number = str(node.value)
        room_text = node.room_type.name.capitalize() + " " + room_number
        text_surface = font.render(room_text, True, (255, 255, 255))
        text_x = rect.x + (rect.width - text_surface.get_width()) // 2
        text_y = rect.y + (rect.height - text_surface.get_height()) // 2
        screen.blit(text_surface, (text_x, text_y))

        # Draw objects in the room
        for obj in room.objects:
            obj.draw(screen)

        # Draw room dimensions
        room_dimensions_text = f"{rect.width:.0f} x {rect.height:.0f}"
        dimensions_surface = font.render(room_dimensions_text, True, (255, 255, 255))
        dimensions_x = rect.x + (rect.width - dimensions_surface.get_width()) // 2
        dimensions_y = text_y + text_surface.get_height()
        screen.blit(dimensions_surface, (dimensions_x, dimensions_y))

    # Draw doors
    for room in rooms:
        if hasattr(room.node, 'door'):
            draw_door(room.node.door, screen)


class Agent(ap.Agent):
    def __init__(self, name, x, y, speed, rooms, graph):
        self.name = name
        self.x = x
        self.y = y
        self.speed = speed
        self.needs = {"energy": 100, "hunger": 100, "thirst": 100, "bladder": 100, "hygiene": 100, "fun": 100}
        self.extra_needs = {"privacy": 100, "groceries": 100, "socialization": 100, "exercise": 100}
        self.target_object = None
        self.color = (155, 155, 255)
        self.radius = 20
        self.all_rooms = rooms
        self.floor_plan_graph = graph
        self.path = []
        self.time_scale_factor_hour = 3600 / 60  # 1 simulated hour
        self.simulated_minute = self.time_scale_factor_hour / 60  # 1 simulated minute
        self.last_update_time = time.time()
        self.interaction_time = None
        self.last_interaction_time = {need: 0 for need in self.needs}
        self.waiting = False
        self.action = "idle"
        self.interaction_start_time = None
        self.interaction_finish_time = None
        self.total_interaction_time_in_minutes = None
        self.exclude_object = None
        self.excluded_objects = {}
        self.paths = []
        self.activities = []
        self.bedroom = None


    def pick_need(self, exclude_needs=None):
        if exclude_needs is None:
            exclude_needs = []

        filtered_needs = {key: value for key, value in self.needs.items() if key not in exclude_needs}
        desired_value = min(list(filtered_needs.values()))
        possible_values = [key for key, value in filtered_needs.items() if value == desired_value]
        #print(possible_values)
        return random.choice(possible_values)

    def pick_object(self, rooms, current_time_in_minutes, exclude_needs=None):
        if exclude_needs is None:
            exclude_needs = []

        need = self.pick_need(exclude_needs)
        satisfying_objects = []

        # add objects
        for room in rooms:
            for obj in room.objects:
                if need in obj.object_type.effect:
                    satisfying_objects.append(obj)

        #handle objects which are being interacted with by other agents
        if len(self.excluded_objects) > 0:
            keys_to_delete = []
            try:
                for object, finish_time in self.excluded_objects.items():
                    if current_time_in_minutes > finish_time:
                        try:
                            keys_to_delete.append(object)
                        except Exception:
                            print("deleting the object caused an issue")
                    elif object in satisfying_objects:
                            try:
                                satisfying_objects.remove(object)
                            except Exception:
                                print(len(satisfying_objects))
                                print(object.name)
                                for o in satisfying_objects:
                                    print(o.name)
                            if len(satisfying_objects) < 1:
                                try:
                                    exclude_needs.append(need)
                                    if len(exclude_needs) < len(self.needs):
                                        self.pick_object(rooms, current_time_in_minutes, exclude_needs )
                                    else:
                                        print("No satisfying objects found for any need.")
                                        self.target_object = None
                                except Exception:
                                    print("there were not enough free objects to satisfy the need, but failed to successfully pick a new one")
                                    print(self.target_object.name)

                for key in keys_to_delete:
                    del self.excluded_objects[key]

            except Exception:
                print("failed to handle object exclusion")
                print(str(current_time_in_minutes))
                print(str(finish_time))
                print("excluded objects: " + str(len(self.excluded_objects)))

        if satisfying_objects:
            self.target_object = random.choice(satisfying_objects)
            self.interaction_time = self.target_object.interaction_time
        else:
            exclude_needs.append(need)
            if len(exclude_needs) < len(self.needs):
                self.pick_object(rooms, current_time_in_minutes, exclude_needs )
            else:
                print("No satisfying objects found for any need.")
                self.target_object = None

    def distance_to_object(self, obj):
        return ((self.x - obj.position[0])**2 + (self.y - obj.position[1])**2)**0.5

    def move_towards_point(self, point, dt):
        self.action = "moving"
        dx = point[0] - self.x
        dy = point[1] - self.y
        distance = ((dx**2) + (dy**2))**0.5

        if distance > self.speed * dt / 1000:
            self.x += dx / distance * self.speed * dt / 1000
            self.y += dy / distance * self.speed * dt / 1000
            return False
        else:
            self.x = point[0]
            self.y = point[1]
            return True

    def interact(self, dt, current_time_in_minutes):
        
        if self.interaction_start_time == None:
            self.interaction_start_time = current_time_in_minutes
            self.total_interaction_time_in_minutes = self.target_object.interaction_time*60
            self.interaction_finish_time = self.total_interaction_time_in_minutes + self.interaction_start_time
            activity = [self.name, " interacting with ", self.target_object.name, " current time in minutes ", str(current_time_in_minutes), " expected interaction finish time ", str(self.interaction_finish_time), " the interaction in hours: ", str(self.target_object.interaction_time)]
            self.activities.append(activity)

        #print(self.name + " interacting with " + self.target_object.name + " |  current time in minutes " + str(current_time_in_minutes) + " |  expected interaction finish time " + str(self.interaction_finish_time) + " |  the interaction in hours: " + str(self.target_object.interaction_time))    
        
        #print("interacting")
        if current_time_in_minutes < self.interaction_finish_time:
            self.action = "interacting"
            self.target_object.interacting_agent = self
            self.interaction_time = self.interaction_finish_time - current_time_in_minutes
            need_increase_per_hour = 5
            need_increase = need_increase_per_hour
            #print(self.name + " interaction time " + str(self.interaction_time) + " with object " + self.target_object.name + " " + str((self.target_object.interaction_time*60)))
            

            for need, effect in self.target_object.object_type.effect.items():
                self.needs[need] += need_increase
                #print(self.needs[need])
                self.needs[need] = min(100, self.needs[need])  # Ensure the need doesn't go above 100
                self.last_interaction_time[need] = time.time()
                #print(str(self.needs[need]) + " is being updated and the value is: " + str(need_increase))

        if current_time_in_minutes >= self.interaction_finish_time:
            self.target_object.interacting_agent = None
            self.target_object = None
            self.interaction_time = None
            self.interaction_start_time = None
            self.interaction_finish_time = None
            self.action = "idle"
            

    def get_room_at_point(self, point):
        for room in self.all_rooms:
            if room.rect.collidepoint(point):
                return room
        return None
    
    def get_room_at_door(self, point):
        rooms = []
        for room in self.all_rooms:
            if room.rect.collidepoint(point):
                rooms.append(room)
        return rooms

    def bfs_shortest_path(self, start_room, goal_room):
        visited = set()
        queue = deque([[start_room]])

        while queue:
            path = queue.popleft()
            current = path[-1]

            if current == goal_room:
                return path

            if current not in visited:
                visited.add(current)

                for neighbor in self.floor_plan_graph[current]:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

        return []

    def draw_needs(self, screen, position_y):
            font = pygame.font.Font(None, 24)
            needs_text = ", ".join(f"{k}: {v}" for k, v in self.needs.items())
            text = font.render(f"{self.name} - {needs_text}", True, (255, 255, 255))
            screen.blit(text, (10, position_y))

    def update(self, rooms, dt, current_time_in_minutes):
        elapsed_time = time.time() - self.last_update_time
        simulated_hour_in_seconds = 60 / 24

        if elapsed_time >= self.simulated_minute:  # Check if one simulated minute has passed   
            self.last_update_time = time.time()  # Reset the timer
            for need, value in self.needs.items():
                # Check if it has been two simulated hours since the last interaction
                if time.time() - self.last_interaction_time[need] >= 2 * simulated_hour_in_seconds:
                    # Randomly decide whether to update this need
                    if random.random() < 0.30:  # 30% chance of updating a need in a simulated minute
                        # Subtract a random value between 1 and 10
                        self.needs[need] -= random.randint(1,10)

                # Ensure the need doesn't go below 0
                self.needs[need] = max(0, self.needs[need])

        if not self.target_object:
            self.pick_object(rooms, current_time_in_minutes, None)
            current_room = self.get_room_at_point((self.x, self.y))
            target_room = self.get_room_at_point(self.target_object.position)

            if current_room and target_room:
                try:
                    room_path = self.bfs_shortest_path(current_room, target_room)
                    assert target_room == room_path[-1]
                    self.paths.append([[e.node.room_type.privacy, e.node.room_type.name] for e in room_path])
                except Exception:
                    print("failed to establish path between rooms")
                    print(target_room)
                    print(room_path[-1])
                # Update the path of doors connecting rooms
                for i in range(len(room_path) - 1):
                    current_room = room_path[i]
                    next_room = room_path[i + 1]

                    door = get_door_between_rooms(current_room, next_room)
                    self.path.append(door.position)

                    """for door in current_room.doors:
                        if next_room in self.get_room_at_door(door.position):
                            self.path.append(door.position)
                            break"""

                self.path.append(self.target_object.position)
                # length of the room_path should be equal to the length of the path
        else:
            if self.distance_to_object(self.target_object) < 1:
                #self.interact(dt, current_time_in_minutes)
                #"""
                if self.target_object.interacting_agent == self or self.target_object.interacting_agent == None:
                    try:
                        self.interact(dt, current_time_in_minutes)
                        #print(self.target_object.interacting_agent.interaction_time)
                    except Exception:
                        print("interaction failed")
                
                elif self.target_object.interacting_agent != None and self.target_object.interacting_agent != self:
                    if self.target_object.interacting_agent.interaction_time <= 10:  # less than 10 minutes of interaction_time left
                        self.action = "waiting"
                        #print(self.name + " is waiting for target object !!!!!" + self.target_object.name +" is being interacted with currently by another agent " + self.target_object.interacting_agent.name + " for duration " + str(self.target_object.interacting_agent.interaction_time))
                        self.waiting = True
                        #print("agent is currently waiting for the object to be free") 
                #"""
                    else:
                        #print(self.name + " trying to interact with object " + self.target_object.name + " object is occupied by " + self.target_object.interacting_agent.name + ", try to do something else")
                        # try to find another object which can satisfy the need, otherwise choose a new need
                        self.excluded_objects[self.target_object] =  self.target_object.interacting_agent.interaction_finish_time
                        self.target_object = None
                        self.interaction_time = None
                        self.interaction_start_time = None
                        self.interaction_finish_time = None
                        self.action = "find new object"
            else:
                if self.path:
                    target = self.path[0]
                    reached_target = self.move_towards_point(target, dt)
                    if reached_target:
                        self.path.pop(0)  # Remove the reached point from the path
                        if not self.path and not self.interaction_time:
                            self.interaction_time = self.target_object.interaction_time


                


    def draw(self, surface):
        # Draw the agent as a circle at its current position
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        #print(self.name + " agent's location is (" + str(self.x) + ", " + str(self.y) + ")") 

        # write the activity of the agent
        task_text = font.render(self.action+ " " + str(self.target_object.name if self.target_object else "picking target object"), True, (0, 0, 0))
        #screen.blit(task_text, (self.x, self.y))
        surface.blit(task_text, (self.x - task_text.get_width() // 2, self.y - task_text.get_height() // 2 + 20))
        #print(self.action+ " " + str(self.target_object.name if self.target_object else "picking target object"))

        # Draw agent's name
        name_text = self.name
        name_surface = font.render(name_text, True, (0, 0, 0))
        surface.blit(name_surface, (self.x - name_surface.get_width() // 2, self.y - name_surface.get_height() // 2 - 20))

        # Draw agent's current destination
        if self.target_object:
            destination_text = f"Dest: {self.target_object.name}"
            destination_surface = font.render(destination_text, True, (0, 0, 0))
            surface.blit(destination_surface, (self.x - destination_surface.get_width() // 2, self.y - destination_surface.get_height() // 2 + 40))



def get_random_room_center(rooms):
    try:
        room = random.choice(rooms)  # Pick a random room from the list of rooms
        center_x = room.rect.x + room.rect.width / 2  # Calculate the x-coordinate of the center
        center_y = room.rect.y + room.rect.height / 2  # Calculate the y-coordinate of the center
        return center_x, center_y
    except Exception:
        print("room could not be picked")
        print(rooms)


def check_room_sizes(rooms):
    values = []
    for room in rooms:
        room.size = (room.rect.width * room.rect.height) / 10000
        print(room.node.room_type.name + " " + str(room.size) + " " + str(room.rect.width) + " " + str(room.rect.height))
        #print(str(room.rect.width) +" vs "+ str(room.node.room_type.min_size[0]))
        #print(str(room.rect.height) +" vs "+ str(room.node.room_type.min_size[1]))
        if room.rect.width >= room.node.room_type.min_size[0] and room.rect.height >= room.node.room_type.min_size[1]:
            values.append("True")
        if room.rect.height >= room.node.room_type.min_size[0] and room.rect.width >= room.node.room_type.min_size[1]:
            values.append("True")
        else:
            values.append("False")

    if "False" in values:
        return False
    else:
        return True



# Set the window size and create the pygame window
screen_width, screen_height = 1500, 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Floorplan Slicing")
pygame.init()
object_font = pygame.font.Font(None, 24)
windows = create_windows(screen)


# Set up the agent font
font = pygame.font.Font(None, 24)
# Set up apartment size
# 80 square meters, 10m x 8m, 1000 x 800, 1m = 100
apartment_width, apartment_height = 1000, 800

pygame.display.flip()
running = True


# when loading the json file, the self.rooms is always calling a function, instead of using the data from the json


class Floorplan():
    def __init__(self, num_rooms, room_types, slicing_tree_root, from_json=False):
        self.num_rooms = num_rooms
        self.room_types = room_types
        self.slicing_tree_root = slicing_tree_root
        self.fitness = None
        self.agents_paths = {}
        self.agents_activites = {}
        self.neighbouring_rooms = {}
        self.mutated = False
        self.crossover = False
        self.parent1 = None
        self.parent2 = None
        self.mutation_probability = None

        if not from_json:
            self.rooms = create_floorplan(slicing_tree_root, pygame.Rect(0, 0, apartment_width, apartment_height))
            self.doors_list = []
            self.graph = {room: [] for room in self.rooms}  # Initialize the graph with empty adjacency lists for each room
            
            identify_neighbors(self.rooms)
            # Print the slicing tree in the console
            #print_tree(slicing_tree_root)

            for room in self.rooms:
                d = add_doors(room, screen, self.doors_list, self.graph)
                if d is not None:
                    self.doors_list, self.graph = d
        else:
            self.rooms = []
            self.doors_list = []
            self.graph = {}
            self.agents_paths = {}
            self.agents_activites = {}
            self.neighbouring_rooms = {}
            self.mutated = False
            self.crossover = False
            self.parent1 = None
            self.parent2 = None
            self.mutation_probability = None
            self.agents_have_bedrooms = None


    def to_dict(self):
        return {
            "num_rooms": self.num_rooms,
            "room_types": [room_type_to_dict(room_type) for room_type in self.room_types],  # Use room_type_to_dict function
            "slicing_tree_root": self.slicing_tree_root.to_dict(),
            "rooms": [room.to_dict() for room in self.rooms],
            "doors_list": [door.to_dict() for door in self.doors_list],
            "graph": [(room.to_dict(), [neighbor.to_dict() for neighbor in neighbors]) for room, neighbors in self.graph.items()],
            "fitness": self.fitness,
            "agents_paths": self.agents_paths,
            "agents_activites": self.agents_activites,
            "neighbouring_rooms": self.neighbouring_rooms,
            "mutated": self.mutated,
            "crossover": self.crossover,
            "parent1": self.parent1,
            "parent2": self.parent2,
            "mutation_probability": self.mutation_probability,
            "agents_have_bedrooms": self.agents_have_bedrooms
        }
    
    @classmethod
    def from_dict(cls, data):
        num_rooms = data['num_rooms']
        room_types = [RoomType(**room_type_data) for room_type_data in data['room_types']]  # Convert dictionaries back to RoomType objects
        slicing_tree_root = SlicingNode.from_dict(data['slicing_tree_root'])
        floorplan_instance = cls(num_rooms, room_types, slicing_tree_root, from_json=True)

        # Load attributes from data dictionary
        floorplan_instance.fitness = data['fitness']
        floorplan_instance.agents_paths = data.get('agents_paths', {})
        floorplan_instance.agents_activites = data.get('agents_activites', {})
        floorplan_instance.neighbouring_rooms = data.get('neighbouring_rooms', {})
        floorplan_instance.mutated = data['mutated']
        floorplan_instance.crossover = data['crossover']
        floorplan_instance.parent1 = data['parent1']
        floorplan_instance.parent2 = data['parent2']
        floorplan_instance.mutation_probability = data['mutation_probability']
        floorplan_instance.agents_have_bedrooms = data['agents_have_bedrooms']

        floorplan_instance.rooms = [Room.from_dict(room_data) for room_data in data['rooms']]
        floorplan_instance.doors_list = [Door.from_dict(door_data) for door_data in data['doors_list']]


        floorplan_instance.graph = {}
        for room_data, neighbors_data in data['graph']:
            room = [room for room in floorplan_instance.rooms if room.node.value == room_data['node']['value']][0]
            neighbors = [room for room in floorplan_instance.rooms if room.node.value in [neighbor['node']['value'] for neighbor in neighbors_data]]
            room.neighbors = neighbors
            floorplan_instance.graph[room] = neighbors
            
        return floorplan_instance








def print_tree(node, level=0):
    if not node:
        return

    if not node.children:  # Leaf node
        print(" " * level * 4 + f"Room {node.value} - {node.room_type.name.capitalize()}")
        return

    print(" " * level * 4 + f"{node.value} ({node.cut_percentage:.2f})")
    for child in node.children:
        print_tree(child, level + 1)


def tree_to_string(node):
    if not node:
        return ""

    if not node.children:  # Leaf node
        return f"R{node.value}{node.room_type.name[:3]}"

    children_strings = [tree_to_string(child) for child in node.children]
    return f"{node.value}{node.cut_percentage:.2f}({children_strings[0]}{children_strings[1]})"


def build_slicing_tree_from_string(tree_string, room_types):
    def find_matching_parenthesis(s):
        stack = []
        for i, c in enumerate(s):
            if c == '(':
                stack.append(i)
            elif c == ')':
                start = stack.pop()
                if not stack:
                    return start, i

    def split_children_string(children_string):
        if "(" not in children_string:
            return children_string, ""

        open_parenthesis, close_parenthesis = find_matching_parenthesis(children_string)
        first_child_string = children_string[:close_parenthesis + 1]
        second_child_string = children_string[close_parenthesis + 1:]
        return first_child_string, second_child_string

    def build_node(tree_string):
        pattern = r'([HV])(\d+\.\d{2})?(?=\(|$)|R(\d+)([a-z]{3})'
    
        root_node = None
        stack = []

        for match in re.finditer(pattern, tree_string):
            node = None

            if match.group(1) is not None and match.group(2) is not None:
                value = match.group(1)
                cut_percentage = match.group(2)

                node = SlicingNode(value, None, None)

                if value in ('H', 'V'):
                    node.cut_percentage = float(cut_percentage)
                    node.children = []
            else:
                room_num = int(match.group(3))
                room_type_code = match.group(4)
                node = SlicingNode(room_num, None, None)
                node.room_type = next(rt for rt in room_types if rt.name[:3] == room_type_code)

            if stack:
                stack[-1].children.append(node)

            if node.value in ('H', 'V'):
                stack.append(node)

            while stack and len(stack[-1].children) == 2:
                completed_node = stack.pop()
                if not stack:
                    root_node = completed_node

        return root_node


    def flatten_children_list(node):
        if not node:
            return

        if isinstance(node.children, list):
            flattened_children = []
            for child in node.children:
                if isinstance(child, list):
                    flattened_children.extend(child)
                elif child is not None:
                    flattened_children.append(child)
            node.children = flattened_children

            for child in node.children:
                flatten_children_list(child)

    root_node = build_node(tree_string)
    #flatten_children_list(root_node)
    return root_node


def generate_population(population_size, num_rooms):
    population = []
    while len(population) < population_size:
        room_types = [random.choice(ROOM_TYPES) for _ in range(num_rooms)]
        slicing_tree_root = generate_slicing_tree(num_rooms, room_types)
        floorplan = Floorplan(num_rooms, room_types, slicing_tree_root)

        # check is floorplan is valid
        if not check_room_sizes(floorplan.rooms):
            print("Room sizes are not good enough, not added to population")
            continue
        tree_string = tree_to_string(slicing_tree_root)
        print(tree_string)
        print_tree(slicing_tree_root)

        population.append(floorplan)
    return population

    
def assign_bedrooms_to_agents(agents, rooms):
    # Create a set to keep track of assigned bedrooms
    assigned_bedrooms = set()
    
    for agent in agents:
        # Skip agents that already have a bedroom assigned
        if agent.bedroom is not None:
            continue
        
        for room in rooms:
            if room.node.room_type.name == "bedroom" and room not in assigned_bedrooms:
                # Assign the room as a bedroom to the agent
                agent.bedroom = room
                assigned_bedrooms.add(room)
                break


def count_agents_with_bedrooms(agents):
    count = 0
    # Add +10 to the count for each agent with a bedroom    
    for agent in agents:
        if agent.bedroom is None:
            count -= 10  
    return count


"""
points for size:
biggest is not living_room: -5
bedroom:
kitchen:
smallest is not bathroom: -5

for room in 
"""


def calculate_fitness(floorplan, screen):
    clock = pygame.time.Clock()
    running = True
    score = 0
    agent_lowest_needs = []
    agents_needs = []
    agents_paths = {}
    agent_activities = {}

    two_private_consecutive = False
    common_private_common = False

    time_scale_factor = 3600 / 60  # 1 hour
    time_scale_factor = 86400 / 60  # 24 hours
    human_speed = 1.8  # meters per second
    scaled_speed = human_speed * time_scale_factor

    elapsed_time = 0
    total_time = 86400
    simulated_hour = 0
    simulated_hour_in_seconds = 60 / 24
    last_hour_time = 0
    remaining_time = 0


    # check is floorplan is valid
    if not check_room_sizes(floorplan.rooms):
        print("Room sizes are not good enough for calculating the fitness")
        #draw_floorplan(floorplan.rooms, screen)
        #pygame.display.update()
        return -1, {}, {}
    
    # TODO if private and social room path
    # agent privacy need

    
    # Create a population of agents
    num_agents = 2
    try:
        agents = [Agent(f"Agent {i+1}", *get_random_room_center(floorplan.rooms), scaled_speed, floorplan.rooms, floorplan.graph) for i in range(num_agents)]

    except Exception:
        print("agents could not be created to calculate fitness")
        print(floorplan.rooms)
        return 0, {}, {}
    

    draw_floorplan(floorplan.rooms, screen)

    for door in floorplan.doors_list:
        draw_door(door, screen)
        pygame.display.update()

    for window in windows:
        for i, room in enumerate(floorplan.rooms):
            if window.colliderect(room.rect):
                room.window = True

    # Perform a room size comparison and window check
    largest_room = max(floorplan.rooms, key=lambda room: room.size)
    smallest_room = min(floorplan.rooms, key=lambda room: room.size)

    for room in floorplan.rooms:
        if room.node.room_type.name == "kitchen":
            if room.window == False:
                if not any(node.node.room_type.privacy == 'common' and node.window for node in room.neighbors): 
                    print("no window in kitchen")
                    score -= 5
        elif room.node.room_type.name == "living_room":
            if room.window == False:
                if not any(node.node.room_type.privacy == 'common' and node.window for node in room.neighbors): 
                    print("no window in living room")
                    score -= 5
            if room != largest_room:
                    score -= 5
        elif room.node.room_type.name == "bedroom":
            if room.window == False:
                print("no window in bedroom")
                score -= 5
        elif room.node.room_type.name == "bathroom":
            if room != smallest_room:
                    score -= 5

    # Check if all agents can have their own bedrooms
    assign_bedrooms_to_agents(agents, floorplan.rooms)
    floorplan.agents_have_bedrooms = count_agents_with_bedrooms(agents)
    score += floorplan.agents_have_bedrooms

    while running:
        dt = clock.tick(60)  # Delta time (in milliseconds)
        #print(dt)
        #elapsed_time += dt * time_scale_factor / 1000  # Convert dt to seconds and scale

        # Calculate and print elapsed and remaining time
        elapsed_time = pygame.time.get_ticks() / 1000  # Convert milliseconds to seconds
        remaining_time = 60 - elapsed_time  # Subtract elapsed time from 60 seconds
        #print("Elapsed time: {:.2f} seconds, Remaining time: {:.2f} seconds".format(elapsed_time, remaining_time))

        # Check if a simulated hour has passed
        if elapsed_time - last_hour_time >= simulated_hour_in_seconds:
            simulated_hour += 1
            last_hour_time = elapsed_time
            #print(f"One simulated hour has passed. It's now hour {simulated_hour}.")
        
        # Calculate simulated minutes within the current hour
        
        simulated_minutes = int((elapsed_time % simulated_hour_in_seconds) * 24)
        current_time_in_minutes = (simulated_hour * 60) + simulated_minutes
        
        # Print the time in the format "hour : minute"
        #print(f"Current time: {simulated_hour}:{simulated_minutes}")

        # Update the agents' states and interactions
        for agent in agents:
            try:
                agent.update(floorplan.rooms, dt, current_time_in_minutes)
            except Exception:
                print("agent could not be updated")
                return 0, {}, {}

        # Clear the screen and draw the updated floor plan
        screen.fill((255, 255, 255))
        draw_floorplan(floorplan.rooms, screen)

        for door in floorplan.doors_list:
            draw_door(door, screen)

            # Draw the windows
        for window in windows:
            pygame.draw.rect(screen, window.color, window)

        # Draw the agents on the screen
        for i, agent in enumerate(agents):
            agent.draw(screen)
            agent.draw_needs(screen, 10 + i * 30)
            pygame.display.update()

        if elapsed_time >= total_time:
            running = False
            break
        if remaining_time < 0:
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    
    for agent in agents:
        agents_paths[agent.name] = agent.paths
        agent_activities[agent.name] = agent.activities
        agent_lowest_needs.append(min(agent.needs.values())) # add the lowest need value (most difficult to fulfill)
        agents_needs.append(agent.needs)

        for row_idx, row in enumerate(agent.paths):
            for path_idx in range(len(row)-1):
                # Check for 'private-private' pattern
                if row[path_idx][0] == 'private' and row[path_idx + 1][0] == 'private':
                    #print(f"In row {row_idx}, paths {path_idx} and {path_idx + 1} have consecutive 'private' values for {agent.name} using {row[path_idx][1]} -> {row[path_idx+1][1]}.")
                    two_private_consecutive = True
            for path_idx in range(len(row) - 2):
                # common - private - common is not good either
                # Check for 'common-private-common' pattern
                if row[path_idx][0] == 'common' and row[path_idx + 1][0] == 'private' and row[path_idx + 2][0] == 'common':
                    #print(f"In row {row_idx}, paths from {path_idx} to {path_idx + 2} form a 'common-private-common' pattern for {agent.name} using {row[path_idx][1]} -> {row[path_idx+1][1]} -> {row[path_idx+2][1]}.")
                    common_private_common = True
    
    if two_private_consecutive:
        score -= 15
    if common_private_common:
        score -= 10
    # check room square meters (common rooms should be biggest, living room, kitchen, bedroom, bathroom)
    # check room connection order for light / windows, between two common areas a wall might not be needed, but between common and public, yes
    # have guests for social interaction, go to work
    
    average_of_needs = sum(agent_lowest_needs)/num_agents
    
    final_score = average_of_needs + score

    if agents_paths == {}:
        return 0, {}, {}

    if average_of_needs == 0:
        counter = 0
        for agent_needs in agents_needs:
            for need, value in agent_needs.items():
                if value == 100:
                    counter += 5
                elif value > 70 and value < 100:
                    counter += 3
                elif value > 50 and value < 70:
                    counter += 1
        final_score = counter + score

    if final_score < 0:
        final_score = 0

    if final_score > 100 and agents_paths != {}:
        print("error, too high score " + str(final_score))
        return 100, {}, {}

    return final_score, agents_paths, agent_activities




def select_chromosomes(population):
    fitness_values = []
    for floorplan in population:
        fitness_values.append(floorplan.fitness)
	
    # normalizes the fitness values to between 0 and 1 and it could be dangerous if too many floorplans have 0 fitness
    fitness_values = [float(i)/sum(fitness_values) for i in fitness_values]
	
    parent1 = random.choices(population, weights=fitness_values, k=1)[0]
    parent2 = random.choices(population, weights=fitness_values, k=1)[0]
	
    print("Selected two floorplans for crossover")
    return parent1, parent2



# crossover on the slicing tree string
# Work in progress

def get_all_hv_nodes(node):
    if not node:
        return {}

    nodes_dict = {}

    if node.value in ('H', 'V'):
        nodes_dict[node] = count_children(node)

    if node.children:
        for child in node.children:
            child_nodes_dict = get_all_hv_nodes(child)
            for child_node, child_count in child_nodes_dict.items():
                if child_node in nodes_dict:
                    nodes_dict[child_node] += child_count
                else:
                    nodes_dict[child_node] = child_count

    return nodes_dict

def count_children(node):
    if not node.children:
        return 0
    
    child_count = 0
    for child in node.children:
        child_count += 1 + count_children(child)
    return child_count

def set_crossover_point(floorplan1, floorplan2):
    all_hv_nodes1 = get_all_hv_nodes(floorplan1)
    all_hv_nodes2 = get_all_hv_nodes(floorplan2)

    if len(all_hv_nodes1) == len(all_hv_nodes2):
        print("hv nodes match")

    # Organize nodes by child count
    nodes_by_child_count1 = organize_nodes_by_child_count(all_hv_nodes1)
    nodes_by_child_count2 = organize_nodes_by_child_count(all_hv_nodes2)

    # Find common child counts
    common_child_counts = set(nodes_by_child_count1.keys()) & set(nodes_by_child_count2.keys())

    if not common_child_counts:
        print("No common child counts found.")
        return None, None

    # Select a random child count
    selected_child_count = random.choice([count for count in common_child_counts if count != max(common_child_counts)])

    # Get nodes with the selected child count
    potential_crossover_nodes1 = nodes_by_child_count1[selected_child_count]
    potential_crossover_nodes2 = nodes_by_child_count2[selected_child_count]

    # Select random crossover indexes from the potential nodes
    crossover_index1 = random.randrange(0, len(potential_crossover_nodes1))
    crossover_index2 = random.randrange(0, len(potential_crossover_nodes2))

    crossover_point1 = potential_crossover_nodes1[crossover_index1]
    crossover_point2 = potential_crossover_nodes2[crossover_index2]

    return crossover_point1, crossover_point2

def organize_nodes_by_child_count(nodes_dict):
    nodes_by_child_count = {}
    for node, count in nodes_dict.items():
        if count not in nodes_by_child_count:
            nodes_by_child_count[count] = []
        nodes_by_child_count[count].append(node)
    return nodes_by_child_count



def crossover(floorplan1, floorplan2):
    for n in range(20):
        # step 1
        tree_string1 = tree_to_string(floorplan1.slicing_tree_root)
        tree_string2 = tree_to_string(floorplan2.slicing_tree_root)
        print("crossover parent 1 : " + tree_string1)
        print("crossover parent 2 : " + tree_string2)

        parent1_constituents = re.findall(r'([HV])|(\d\.\d{2})|(R\d\w{3})|([\(\)])', tree_string1)
        parent1_constituents = [c for tup in parent1_constituents for c in tup if c != '']
        parent1_room_count = sum(1 for const in parent1_constituents if const and const.startswith('R'))
        
        # step 2 crossover
        crossover_point1, crossover_point2 = set_crossover_point(floorplan1.slicing_tree_root, floorplan2.slicing_tree_root)
        formatted_cut_percentage = "{:.2f}".format(crossover_point1.cut_percentage)
        crossover_index1 = tree_string1.find(str(crossover_point1.value) + formatted_cut_percentage)
        formatted_cut_percentage = "{:.2f}".format(crossover_point2.cut_percentage)
        crossover_index2 = tree_string2.find(str(crossover_point2.value) + formatted_cut_percentage)
        
        child_tree_string1 = tree_string1[:crossover_index1] + tree_string2[crossover_index2:]
        print("crossover result : " + child_tree_string1)

        constituents = re.findall(r'([HV])|(\d\.\d{2})|(R\d\w{3})|([\(\)])', child_tree_string1)
        constituents = [c for tup in constituents for c in tup if c != '']
        room_count = sum(1 for const in constituents if const and const.startswith('R'))
        print("crossover floorplan's room count: " + str(room_count))
        
        if room_count != parent1_room_count:
            print("crossover floorplan and original floorplan do not have the same amount of rooms")
            continue

        child1 = build_slicing_tree_from_string(child_tree_string1, ROOM_TYPES)

        child = Floorplan(room_count, ROOM_TYPES, child1)

        if check_room_sizes(child.rooms):
            print("Performed crossover between two chromosomes")
            return child
                
    return child

        #input_tree_string = "H0.66(H0.58(R2batR3bed)V0.36(R1bedR4liv))"
        #root_node = build_slicing_tree_from_string(input_tree_string, ROOM_TYPES)
        #print_tree(root_node)









#mutated floorplan by changing a cut location test
"""
transformed_list = ['H0.54(H0.53(R2kitR1liv)V0.76(R3bedR4liv))', 'H0.54(V0.76(H0.53(R2kitR1liv))R3bedR4liv)', 'V0.75(R2bedH0.24(H0.61(R1livR3kit)R4kit))', 'V0.75(H0.24(R2bed)H0.61(R1livR3kit)R4kit)', 'V0.45(H0.53(R4bedV0.75(R1bedR3kit))R2kit)', 'V0.45(H0.53(V0.75(R4bed)R1bedR3kitR2kit))', 'H0.30(R4kitH0.74(R2kitV0.66(R3livR1liv)))', 'H0.30(V0.66(R4kit)H0.74(R2kitR3livR1liv))', 'V0.63(H0.79(R1livR3liv)V0.28(R2bedR4liv))', 'V0.63(H0.79(R1liv)R3livV0.28(R2bedR4liv))', 'V0.57(H0.39(R3bedR1kit)V0.66(R4livR2kit))', 'V0.57(H0.39(R3bed)R1kitV0.66(R4livR2kit))', 'H0.56(V0.66(R3livR2kit)H0.38(R4kitR1kit))', 'H0.56(V0.66(R3liv)R2kitH0.38(R4kitR1kit))', 'H0.41(H0.59(R4kitR1kit)V0.35(R2livR3bed))', 'H0.59(H0.41(R4kitR1kitV0.35(R2livR3bed)))', 'H0.46(H0.43(V0.30(R2bedR1liv)R4bed)R3bed)', 'V0.30(H0.46(H0.43(R2bedR1livR4bed)R3bed))', 'H0.63(V0.34(R4kitR2bed)H0.74(R3kitR1kit))', 'H0.63(V0.34(H0.74(R4kit)R2bed)R3kitR1kit)', 'H0.48(H0.59(V0.24(R1bedR2liv)R4bed)R3liv)', 'H0.59(H0.48(V0.24(R1bedR2liv)R4bedR3liv))', 'V0.35(H0.41(V0.34(R3kitR1kit)R2bed)R4kit)', 'V0.35(H0.41(V0.34(R3kit)R1kitR2bed)R4kit)', 'H0.43(H0.53(R4livR2kit)V0.79(R1bedR3kit))', 'H0.43(H0.53(R4liv)R2kitV0.79(R1bedR3kit))', 'V0.54(H0.59(R4kitV0.63(R1bedR3kit))R2bed)', 'V0.54(H0.59(V0.63(R4kit)R1bedR3kitR2bed))', 'H0.21(H0.63(R2kitR4kit)V0.24(R3bedR1bed))', 'H0.21(H0.63(R2kit)R4kitV0.24(R3bedR1bed))', 'H0.68(V0.67(R1livH0.24(R2bedR4liv))R3liv)', 'H0.24(H0.68(V0.67(R1livR2bedR4livR3liv)))', 'H0.52(H0.67(H0.77(R4livR1kit)R2kit)R3liv)', 'H0.52(H0.67(H0.77(R4livR1kit))R2kitR3liv)', 'V0.28(V0.21(V0.64(R2bedR3kit)R4liv)R1liv)', 'V0.21(V0.28(V0.64(R2bedR3kit)R4livR1liv))', 'H0.65(V0.40(R3bedR4bed)H0.73(R1livR2bed))', 'H0.65(V0.40(H0.73(R3bed)R4bed)R1livR2bed)', 'V0.70(H0.35(R3livV0.23(R4bedR1liv))R2liv)', 'V0.70(H0.35(V0.23(R3liv)R4bedR1livR2liv))', 'H0.45(V0.20(R2livR1liv)V0.46(R3bedR4kit))', 'H0.45(V0.20(R2liv)R1livV0.46(R3bedR4kit))', 'V0.51(R1kitH0.38(V0.42(R3kitR2bed)R4kit))', 'V0.51(H0.38(R1kit)V0.42(R3kitR2bed)R4kit)', 'V0.52(V0.57(R2bedR3bed)H0.73(R4livR1bed))', 'V0.52(V0.57(R2bed)R3bedH0.73(R4livR1bed))', 'V0.67(H0.59(H0.38(R3kitR1liv)R4kit)R2bed)', 'V0.67(H0.38(H0.59(R3kit)R1liv)R4kitR2bed)', 'H0.42(H0.33(R2kitV0.78(R4kitR3bed))R1kit)', 'H0.42(H0.33(R2kit))', 'V0.31(V0.29(R3kitR2bed)H0.29(R4kitR1kit))', 'V0.31(V0.29(R3kit)R2bedH0.29(R4kitR1kit))', 'H0.58(H0.66(R3kitR4liv)V0.60(R2livR1liv))', 'V0.60(H0.58(H0.66(R3kitR4liv)R2livR1liv))', 'V0.35(H0.26(R1bedR2kit)V0.25(R4bedR3bed))', 'V0.35(H0.26(V0.25(R1bed)R2kit)R4bedR3bed)', 'V0.44(H0.28(R1kitR2liv)H0.28(R4livR3liv))', 'V0.44(H0.28(R1kit)R2livH0.28(R4livR3liv))', 'H0.47(R4kitV0.46(H0.57(R2kitR1kit)R3liv))', 'H0.47(V0.46(R4kit)H0.57(R2kitR1kit)R3liv)', 'V0.57(V0.39(R2bedR3kit)H0.45(R4kitR1kit))', 'V0.57(V0.39(H0.45(R2bed)R3kit)R4kitR1kit)', 'V0.77(V0.80(R1kitR3bed)V0.32(R4kitR2kit))', 'V0.77(V0.80(R1kit)R3bedV0.32(R4kitR2kit))', 'V0.72(V0.79(H0.60(R2kitR3liv)R1kit)R4kit)', 'V0.72(H0.60(V0.79(R2kit)R3liv)R1kitR4kit)', 'H0.66(H0.65(V0.61(R2bedR3kit)R4liv)R1kit)', 'H0.66(H0.65(V0.61(R2bedR3kit))R4livR1kit)', 'V0.44(H0.45(V0.25(R1kitR4kit)R2bed)R3kit)', 'V0.44(H0.45(V0.25(R1kit)R4kitR2bed)R3kit)', 'V0.73(R1bedV0.22(H0.65(R2livR3kit)R4kit))', 'V0.73(H0.65(R1bed)V0.22(R2livR3kitR4kit))', 'V0.77(R4livH0.38(V0.21(R1bedR3liv)R2kit))', 'V0.77(H0.38(R4liv)V0.21(R1bedR3liv)R2kit)', 'V0.57(H0.59(R1livR3bed)H0.29(R4kitR2kit))', 'V0.57(H0.59(R1liv)R3bedH0.29(R4kitR2kit))', 'V0.44(H0.65(R1kitR3liv)V0.30(R2kitR4liv))', 'V0.44(H0.65(V0.30(R1kit)R3liv)R2kitR4liv)', 'V0.65(R3bedV0.31(R1kitH0.38(R2bedR4bed)))', 'V0.65(H0.38(R3bed)V0.31(R1kitR2bedR4bed))', 'H0.58(H0.61(R4kitH0.50(R3kitR1kit))R2kit)', 'H0.58(H0.61(R4kit))', 'V0.70(R2kitH0.60(R4kitV0.64(R1kitR3kit)))', 'V0.70(H0.60(R2kit)R4kitV0.64(R1kitR3kit))', 'H0.38(V0.73(H0.20(R4livR1liv)R3bed)R2kit)', 'H0.38(V0.73(H0.20(R4livR1liv))R3bedR2kit)', 'V0.41(R4bedH0.70(V0.46(R3livR2liv)R1kit))', 'V0.41(V0.46(R4bed)H0.70(R3livR2livR1kit))', 'H0.34(R1bedH0.73(V0.23(R4bedR3liv)R2bed))', 'H0.34(H0.73(R1bed)V0.23(R4bedR3liv)R2bed)', 'V0.21(V0.30(R2livR1kit)H0.59(R4bedR3kit))', 'V0.30(V0.21(R2livR1kitH0.59(R4bedR3kit)))', 'V0.26(H0.48(R1livR3bed)H0.30(R2bedR4kit))', 'H0.48(V0.26(R1livR3bedH0.30(R2bedR4kit)))', 'V0.59(V0.75(R2kitH0.45(R3kitR1kit))R4liv)', 'V0.59(H0.45(V0.75(R2kitR3kitR1kitR4liv)))', 'H0.70(R2livV0.77(R3bedV0.46(R1kitR4liv)))', 'H0.70(V0.77(R2liv)R3bedV0.46(R1kitR4liv))']

transformed_list = ['V0.45(R4bedV0.75(R1bedR3kit)R2kitH0.53)', 'V0.45(V0.75(H0.53(R4bedR1bedR3kitR2kit)))', 'V0.45(H0.53(V0.75(R4bed)R1bedR3kitR2kit))']

for list in transformed_list:
    print(list)
    a = build_slicing_tree_from_string(list, ROOM_TYPES)
    room_types = [random.choice(ROOM_TYPES) for _ in range(num_rooms)]
    floorplan = Floorplan(4, room_types, a)
    draw_floorplan(floorplan.rooms, screen)
    pygame.display.update()

room_types = [random.choice(ROOM_TYPES) for _ in range(num_rooms)]
string_list = []
for i in range(1,50):
    slicing_tree_root = generate_slicing_tree(4, room_types)
    strng = tree_to_string(slicing_tree_root)
    string_list.append(strng)

string_list

"""

"""
# draw plan from slicing treee
a = build_slicing_tree_from_string('V0.59(H0.71(R3kitR1liv)H0.31(R4bedH0.42(R5bedR2bat)))', ROOM_TYPES)
room_types = [random.choice(ROOM_TYPES) for _ in range(5)]
floorplan = Floorplan(4, room_types, a)
draw_floorplan(floorplan.rooms, screen)
pygame.display.update()
"""
def mutate_parse_tree_string(tree_string):
    stack = []
    root = []
    stack.append(root)
    for token in re.findall(r'([HV]\d+\.\d{2})|(R\d+[a-z]{3})', tree_string):
        if token[0]: # token[0] will be non-empty if it's a cut (leaf)
            current_node = [token[0], []]
            stack[-1].append(current_node)
            stack.append(current_node[1])
        elif token[1]: # token[1] will be non-empty if it's a room (node)
            current_node = [token[1], []]
            stack[-1].append(current_node)
            # No need to push this onto the stack as it won't have its own children.
        if tree_string[tree_string.find(token[0] if token[0] else token[1]) + len(token[0] if token[0] else token[1]) ] == ')':
            stack.pop()
    return root

def mutate_get_all_cuts(tree, parent=None):
    cuts = []
    for node in tree:
        if node[1]:  # if the node has children, it's a cut
            cuts.append((node, parent))
            cuts.extend(mutate_get_all_cuts(node[1], node))
    return cuts

def mutate_remove_cut(tree, cut_to_remove, parent=None):
    for i, node in enumerate(tree):
        if node is cut_to_remove:
            tree.pop(i)
            for child in reversed(node[1]): # reinsert all children
                tree.insert(i, child)
            return True
        if node[1] and mutate_remove_cut(node[1], cut_to_remove, node):
            return True
    return False

def mutate_insert_cut(tree, cut_to_insert):
    for i, node in enumerate(tree):
        if node[1]:  # if the node has children, it's a cut
            # Only consider cuts with one or two children.
            if len(node[1]) < 3:
                if len(node[1]) == 2 and random.random() < 0.2:
                    # If the cut has two children, randomly select one.
                    index = random.choice(range(len(node[1])))
                else:
                    # If the cut has only one child, select it.
                    index = 0
                # Insert the cut to be inserted as a parent to the selected child.
                cut_to_insert[1].append(node[1][index].copy())
                node[1][index] = cut_to_insert
                return True
            # Try to insert the cut in the children of the current cut.
            if mutate_insert_cut(node[1], cut_to_insert):
                return True
    return False

def mutate_transform_tree(tree_string):
    tree = mutate_parse_tree_string(tree_string)
    cuts = mutate_get_all_cuts(tree)
    cut_to_remove = random.choice(cuts[1:])  # choose a cut to remove, not the root
    mutate_remove_cut(tree, cut_to_remove[0])
    cut_to_remove[0][1] = []  # clear children of the cut to be reinserted
    if not mutate_insert_cut(tree, cut_to_remove[0]):  # try to insert the cut
        tree[0][1].append(cut_to_remove[0])  # if cannot insert anywhere, append to root
    if not tree[0][1]:  # if root is a room, append root to the cut to remove
        cut_to_remove[0][1].append(tree[0].copy())
        tree[0][0], tree[0][1] = cut_to_remove[0][0], cut_to_remove[0][1]
    return tree

def mutate_tree_to_string(tree):
    result = ''
    for node in tree:
        if node[1]:  # if the node has children, it's a cut
            result += node[0] + '(' + mutate_tree_to_string(node[1]) + ')'
        else:  # if the node doesn't have children, it's a room
            result += node[0]
    return result

def mutate_verify_tree(tree_string, transformed_tree_string):
    # Check if the two strings are identical
    if tree_string == transformed_tree_string:
        print("The original tree string and transformed tree string are identical.")
        return False

    original_cuts = re.findall(r'([HV]\d+\.\d{2})', tree_string)
    original_rooms = re.findall(r'(R\d+[a-z]{3})', tree_string)

    transformed_cuts = re.findall(r'([HV]\d+\.\d{2})', transformed_tree_string)
    transformed_rooms = re.findall(r'(R\d+[a-z]{3})', transformed_tree_string)

    # Check if the numbers of cuts and rooms are the same in the original and transformed strings
    if len(original_cuts) != len(transformed_cuts) or len(original_rooms) != len(transformed_rooms):
        print("The number of cuts or rooms in the original and transformed strings are not the same.")
        return False

    return True


def swap_cuts(tree_string):
    # Use regex to find all cuts
    cuts = [(m.start(), m.end(), m.group()) for m in re.finditer(r'[HV]\d+\.\d+', tree_string)]
    
    # If there's less than 2 cuts, return the original string
    if len(cuts) < 2:
        return tree_string

    # Randomly select two different cuts
    cut1, cut2 = random.sample(cuts, 2)

    # Create a new tree string by piecing together the parts before, between, and after the cuts, with the cuts swapped
    start1, end1, value1 = cut1
    start2, end2, value2 = cut2
    
    if start1 < start2:
        new_tree_string = tree_string[:start1] + value2 + tree_string[end1:start2] + value1 + tree_string[end2:]
    else:
        new_tree_string = tree_string[:start2] + value1 + tree_string[end2:start1] + value2 + tree_string[end1:]

    return new_tree_string

def mutate_floorplan(floorplan, mutation_rate):
    try:
        transformed_tree_string = None
        # Convert the tree to a string representation
        tree_string = tree_to_string(floorplan.slicing_tree_root) #"H0.66(H0.58(R2batR3bed)V0.36(R1bedR4liv))"
        tree_string_copy = tree_string
        #print("tree_string:")
        #print(tree_string)

        # Split the tree_string into a list of its constituents: 'H', 'V', room_types, cut_percentages, and parentheses
        original_constituents_find = re.findall(r'([HV])|(\d\.\d{2})|(R\d\w{3})|([\(\)])', tree_string)
        constituents_find = re.findall(r'([HV])|(\d\.\d{2})|(R\d\w{3})|([\(\)])', tree_string)

        # Flatten the list of tuples and remove empty strings
        constituents = [c for tup in constituents_find for c in tup if c != '']
        room_count = sum(1 for const in constituents if const and const.startswith('R'))
        #print("mutated floorplan's room count: ")
        #print(room_count)

        # Perform the mutations independently
        
        # Mutation operator 1: mutate 'H' or 'V' by changing values
        if mutation_rate > random.random():
            try:
                hv_indices = [i for i, const in enumerate(constituents) if const in ['H', 'V']]
                if hv_indices:  # if list is not empty
                    idx = random.choice(hv_indices)
                    constituents[idx] = 'V' if constituents[idx] == 'H' else 'H'
                    tree_string = ''.join(constituents)
                    constituents_find = re.findall(r'([HV])|(\d\.\d{2})|(R\d\w{3})|([\(\)])', tree_string)
                    constituents = [c for tup in constituents_find for c in tup if c != '']
                    print("after changing H-V values: " + tree_string)
            except Exception:
                print("HV value change failed")
                constituents = [c for tup in constituents_find for c in tup if c != '']

        # Mutation operator 1: mutate 'H' or 'V' by swapping values
        if mutation_rate > random.random():
            try:
                new_tree_string = swap_cuts(tree_string)
                tree_string = new_tree_string
                constituents_find = re.findall(r'([HV])|(\d\.\d{2})|(R\d\w{3})|([\(\)])', tree_string)
                constituents = [c for tup in constituents_find for c in tup if c != '']
                print("after swapping H-V values " + tree_string)
            except Exception:
                print("HV value swap failed")
                constituents = [c for tup in constituents_find for c in tup if c != '']
                
        # Mutation operator 1: mutate 'H' or 'V'by changing cut location
        #"H0.66(H0.58(R2batR3bed)V0.36(R1bedR4liv))"
        if mutation_rate > random.random():
            try:
                transformed_tree = mutate_transform_tree(tree_string)
                transformed_tree_string = mutate_tree_to_string(transformed_tree)
                print("tree before cut location mutation: " + tree_string)
                print("transformed tree built from mutation: " + transformed_tree_string)
                temp_tree = build_slicing_tree_from_string(transformed_tree_string, ROOM_TYPES)
                temp_tree_string = tree_to_string(temp_tree)
                print("temp tree string built from mutation after building a tree: " + temp_tree_string)
                temp_constituents = re.findall(r'([HV])|(\d\.\d{2})|(R\d\w{3})|([\(\)])', temp_tree_string)
                #print(len(temp_constituents) == len(constituents))
                if len(temp_constituents) == len(constituents):
                    if mutate_verify_tree(tree_string, temp_tree_string):
                        verify_tree_root = build_slicing_tree_from_string(temp_tree_string, ROOM_TYPES)
                        verify_floorplan = Floorplan(room_count, ROOM_TYPES, verify_tree_root)
                        if check_room_sizes(verify_floorplan.rooms):
                            tree_string = temp_tree_string
                            constituents_find = re.findall(r'([HV])|(\d\.\d{2})|(R\d\w{3})|([\(\)])', tree_string)
                            constituents = [c for tup in constituents_find for c in tup if c != '']
                            print("after changing H-V cut :" + tree_string)
            except Exception:
                print("cut location change failed")
                constituents = [c for tup in constituents_find for c in tup if c != '']

        # Mutation operator 2: mutate cut_percentage
        """
        if mutation_rate > random.random():
            try:
                cut_indices = [i for i, const in enumerate(constituents) if re.match(r'\d\.\d+', const)]
                if cut_indices:  # if list is not empty
                    idx = random.choice(cut_indices)
                    new_cut_percentage = "{:.2f}".format(round(random.uniform(0.35, 0.65), 2))
                    constituents[idx] = new_cut_percentage
                    tree_string = ''.join(constituents)
                    #print(tree_string)
                    constituents_find = re.findall(r'([HV])|(\d\.\d{2})|(R\d\w{3})|([\(\)])', tree_string)
                    constituents = [c for tup in constituents_find for c in tup if c != '']
                    #print("after changing H-V cut percentage")
                    #print(tree_string)
            except Exception:
                print("cut value change failed")
                constituents = [c for tup in constituents_find for c in tup if c != '']
        """
        # Mutation operator 3: mutate room_type
        if mutation_rate > random.random():
            try:
                room_indices = [i for i, const in enumerate(constituents) if re.match(r'R\d\w+', const)]
                if room_indices:  # if list is not empty
                    idx = random.choice(room_indices)
                    room_number = constituents[idx][1]  # get the room number
                    room_type = random.choice(ROOM_TYPES)
                    new_room_type = next(rt.name[:3] for rt in ROOM_TYPES if rt == room_type)
                    constituents[idx] = 'R' + room_number + str(new_room_type)
                    tree_string = ''.join(constituents)
                    constituents_find = re.findall(r'([HV])|(\d\.\d{2})|(R\d\w{3})|([\(\)])', tree_string)
                    constituents = [c for tup in constituents_find for c in tup if c != '']
                    print("after changing room types:" + tree_string)
            except Exception:
                print("room value change failed")
                constituents = [c for tup in constituents_find for c in tup if c != '']

        # Recombine the mutated constituents back into a tree_string and convert it back to a tree
        mutated_tree_string = ''.join(constituents)
        #print("mutated_tree_string")
        #print(mutated_tree_string)

    # Convert the mutated tree string back to a SlicingNode structure

        mutated_tree_root = build_slicing_tree_from_string(mutated_tree_string, ROOM_TYPES)
        floorplan = Floorplan(room_count, ROOM_TYPES, mutated_tree_root)

        # Update the floorplan's slicing_tree_root with the mutated one
        #floorplan.slicing_tree_root = mutated_tree_root
        #floorplan.num_rooms = room_count

        if floorplan is not None:
            return floorplan
    except Exception:
        print("mutation failed")

"""
# TEST MUTATION WITH NEW FUNCTIONS
tree_list = ['H0.54(H0.53(R2kitR1liv)V0.76(R3bedR4liv))', 'V0.75(R2bedH0.24(H0.61(R1livR3kit)R4kit))', 'V0.45(H0.53(R4bedV0.75(R1bedR3kit))R2kit)', 'H0.30(R4kitH0.74(R2kitV0.66(R3livR1liv)))', 'V0.63(H0.79(R1livR3liv)V0.28(R2bedR4liv))', 'V0.57(H0.39(R3bedR1kit)V0.66(R4livR2kit))', 'H0.56(V0.66(R3livR2kit)H0.38(R4kitR1kit))', 'H0.41(H0.59(R4kitR1kit)V0.35(R2livR3bed))', 'H0.46(H0.43(V0.30(R2bedR1liv)R4bed)R3bed)', 'H0.63(V0.34(R4kitR2bed)H0.74(R3kitR1kit))', 'H0.48(H0.59(V0.24(R1bedR2liv)R4bed)R3liv)', 'V0.35(H0.41(V0.34(R3kitR1kit)R2bed)R4kit)', 'H0.43(H0.53(R4livR2kit)V0.79(R1bedR3kit))', 'V0.54(H0.59(R4kitV0.63(R1bedR3kit))R2bed)', 'H0.21(H0.63(R2kitR4kit)V0.24(R3bedR1bed))', 'H0.68(V0.67(R1livH0.24(R2bedR4liv))R3liv)', 'H0.52(H0.67(H0.77(R4livR1kit)R2kit)R3liv)', 'V0.28(V0.21(V0.64(R2bedR3kit)R4liv)R1liv)', 'H0.65(V0.40(R3bedR4bed)H0.73(R1livR2bed))', 'V0.70(H0.35(R3livV0.23(R4bedR1liv))R2liv)', 'H0.45(V0.20(R2livR1liv)V0.46(R3bedR4kit))', 'V0.51(R1kitH0.38(V0.42(R3kitR2bed)R4kit))', 'V0.52(V0.57(R2bedR3bed)H0.73(R4livR1bed))', 'V0.67(H0.59(H0.38(R3kitR1liv)R4kit)R2bed)', 'H0.42(H0.33(R2kitV0.78(R4kitR3bed))R1kit)', 'V0.31(V0.29(R3kitR2bed)H0.29(R4kitR1kit))', 'H0.58(H0.66(R3kitR4liv)V0.60(R2livR1liv))', 'V0.35(H0.26(R1bedR2kit)V0.25(R4bedR3bed))', 'V0.44(H0.28(R1kitR2liv)H0.28(R4livR3liv))', 'H0.47(R4kitV0.46(H0.57(R2kitR1kit)R3liv))', 'V0.57(V0.39(R2bedR3kit)H0.45(R4kitR1kit))', 'V0.77(V0.80(R1kitR3bed)V0.32(R4kitR2kit))', 'V0.72(V0.79(H0.60(R2kitR3liv)R1kit)R4kit)', 'H0.66(H0.65(V0.61(R2bedR3kit)R4liv)R1kit)', 'V0.44(H0.45(V0.25(R1kitR4kit)R2bed)R3kit)', 'V0.73(R1bedV0.22(H0.65(R2livR3kit)R4kit))', 'V0.77(R4livH0.38(V0.21(R1bedR3liv)R2kit))', 'V0.57(H0.59(R1livR3bed)H0.29(R4kitR2kit))', 'V0.44(H0.65(R1kitR3liv)V0.30(R2kitR4liv))', 'V0.65(R3bedV0.31(R1kitH0.38(R2bedR4bed)))', 'H0.58(H0.61(R4kitH0.50(R3kitR1kit))R2kit)', 'V0.70(R2kitH0.60(R4kitV0.64(R1kitR3kit)))', 'H0.38(V0.73(H0.20(R4livR1liv)R3bed)R2kit)', 'V0.41(R4bedH0.70(V0.46(R3livR2liv)R1kit))', 'H0.34(R1bedH0.73(V0.23(R4bedR3liv)R2bed))', 'V0.21(V0.30(R2livR1kit)H0.59(R4bedR3kit))', 'V0.26(H0.48(R1livR3bed)H0.30(R2bedR4kit))', 'V0.59(V0.75(R2kitH0.45(R3kitR1kit))R4liv)', 'H0.70(R2livV0.77(R3bedV0.46(R1kitR4liv)))']
mutation_list = []

for tree in tree_list:
    #transformed_tree = mutate_transform_tree(tree)
    #transformed_tree_string = mutate_tree_to_string(transformed_tree)
    #print(tree)
    #print(transformed_tree_string)
    a = build_slicing_tree_from_string(tree, ROOM_TYPES)
    room_types = [random.choice(ROOM_TYPES) for _ in range(num_rooms)]
    floorplan = Floorplan(4, room_types, a)
    draw_floorplan(floorplan.rooms, screen)
    pygame.display.update()
    floorplan = mutate_floorplan(floorplan, 0.9)
    draw_floorplan(floorplan.rooms, screen)
    pygame.display.update()
    new_tree = tree_to_string(floorplan.slicing_tree_root)
    #print(new_tree)
    mutation_list.append(new_tree)

mutation_list
"""


def run_simulation(params):
    try:
        print(f"Running simulation with params: {params}")
        plan, chunk_index, i, it_counter = params
        # Set the window size and create the pygame window
        screen_width, screen_height = 1500, 800
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"Floorplan {i+1}")
        pygame.init()

        # Calculate fitness
        fitness, plan.agents_paths, plan.agents_activites = calculate_fitness(plan, screen)
        
        if fitness >= 0:

            plan.fitness = fitness
            print(plan.fitness)

            for room in plan.rooms:
                plan.neighbouring_rooms[room.node.room_type.name] = [node.node.room_type.name for node in room.neighbors]

            # Save the floorplan
            pygame.image.save(screen,f"D:\\ProgrammingF#\\Master's thesis\\generation0\\sample_{it_counter}_{chunk_index}_{i}_{plan.fitness}.jpg")
            floorplan_dict = plan.to_dict()
            filename = f"D:\\ProgrammingF#\\Master's thesis\\generation0\\sample_{it_counter}_{chunk_index}_{i}_{plan.fitness}.json"
            with open(filename, "w") as outfile:
                json.dump(floorplan_dict, outfile)
                print("created json object")
    except Exception:
        print("failed to run simulation")


def chunks(lst, n):
    "Yield successive n-sized chunks from list."
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



#OPEN SPECIFIC FLOORPLAN FOR TEST
"""
with open("D:\ProgrammingF#\Master's thesis\population0\sample_15_0_109.0.json", "r") as infile:
    floorplan2 = json.load(infile)

floorplan2 = Floorplan.from_dict(floorplan2)

calculate_fitness(floorplan2, screen)

print("stop")
"""


"""

"""


def plot_fitness_values(directory):
    fitness_values = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                floorplan = Floorplan.from_dict(data)
                fitness_values.append(floorplan.fitness)

    # Plot the fitness values
    plt.plot(fitness_values)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.show()

# plot_fitness_values("D:\\ProgrammingF#\\Master's thesis\\population0")



def create_dataframe_from_json(directory):
    # List to store data dictionaries
    data_list = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                floorplan = Floorplan.from_dict(data)
                
                # Create a dictionary with all the required values
                data_dict = {
                    'num_rooms': floorplan.num_rooms,
                    'slicing_tree_root': floorplan.slicing_tree_root,
                    'floorplan_instance': floorplan,
                    'fitness': floorplan.fitness,
                    'rooms': floorplan.rooms,
                    'doors_list': floorplan.doors_list,
                    'tree_string': tree_to_string(floorplan.slicing_tree_root)  # assuming tree_to_string is a function that takes a slicing_tree_root as input and returns a string
                }

                # Append the dictionary to the data list
                data_list.append(data_dict)
    
    # Convert the data list to a pandas DataFrame
    df = pd.DataFrame(data_list)
    
    return df

# df = create_dataframe_from_json("D:\\ProgrammingF#\\Master's thesis\\population0")

def open_generation(directory):
    population = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                floorplan = Floorplan.from_dict(data)
                population.append(floorplan)
    return population




def run_mutated_simulation(params):
    try:
        child, i, chunk_index, j, num_of_rooms, folder_path = params

        screen_width, screen_height = 1500, 800
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"Floorplan {i} - {j}")
        pygame.init()

        try:
            # Calculate fitness
            fitness, child.agents_paths, child.agents_activites = calculate_fitness(child, screen)
            child.fitness = fitness
        except Exception:
            print("fitness calculation failed")

        try:
            # Save the floorplan
            floorplan_dict = child.to_dict()
            filename = folder_path + f"generation_{i}_{chunk_index}_{j}_{child.fitness}.json"
            print("Saving the floorplan json " + filename)
            try:
                with open(filename, "w") as outfile:
                    json.dump(floorplan_dict, outfile)
                    print("Created json object " + filename)
            except Exception:
                print("Could not save the floorplan json " + filename)
            print("Saving the floorplan picture " + f"generation_{i}_{chunk_index}_{j}_{child.fitness}.jpg")
            pygame.image.save(screen, folder_path + f"generation_{i}_{chunk_index}_{j}_{child.fitness}.jpg")
        except Exception:
            print(f"----------------> Could not save floorplan, params: {chunk_index}, {j} {child.fitness}, {num_of_rooms}, " + folder_path + f"generation_{i}_{chunk_index}_{j}_{child.fitness}.jpg")

    except Exception:
        print("simulation run failed")

def mutate_or_crossover(params):
    try:
        floorplan1, floorplan2, num_of_rooms = params
        mutation_probability = 1
        crossover_probability = 0
        
        child = floorplan1 if random.random() < 0.5 else floorplan2

        try:
            if random.random() < crossover_probability:
                child = crossover(floorplan1, floorplan2)
                child.crossover = True
                child.parent1 = tree_to_string(floorplan1.slicing_tree_root)
                child.parent2 = tree_to_string(floorplan2.slicing_tree_root)
        except Exception:
            print("Could not process crossover")

        try:
            if random.random() < mutation_probability:
                child = mutate_floorplan(child, mutation_probability)
                child.mutated = True
                child.mutation_probability = mutation_probability
                child.parent1 = tree_to_string(floorplan1.slicing_tree_root)
                child.parent2 = tree_to_string(floorplan2.slicing_tree_root)
        except Exception:
            print("Could not process mutation")

        check_string = tree_to_string(child.slicing_tree_root)
        constituents = re.findall(r'([HV])|(\d\.\d{2})|(R\d\w{3})|([\(\)])', check_string)
        constituents = [c for tup in constituents for c in tup if c != '']
        child.num_rooms = sum(1 for const in constituents if const and const.startswith('R'))

        if child.slicing_tree_root is not None and check_room_sizes(child.rooms) and child.num_rooms == num_of_rooms:
            return child

    except Exception:
        print("Could not process the parameters")




# THIS IS THE MAIN WAY OF GENERATING A POPULATION

# PLEASE SET THE FOLDER PATH TO SAVE THE JSONS
# Please set the target_count to match the amount of floorplans in the total population of the initial generation
# target_count = 10
# Please change the value of 50 if it does not match the amount you would like to generate in each iteration 
# population = generate_population(50, num_of_rooms)
# Please change the chunks if you would like to process more in a single iteration, I usually make enough chunks to match the Pool size
# population_chunks = list(chunks(population, 10))
# In Pool(10), 10 can be swapped with any amount and that will initiate that amount of multiprocessing pools
# Pool(10)

num_of_rooms = 5

if __name__ == '__main__':
    it_counter = 0

    folder_path = r"D:\ProgrammingF#\Master's thesis\generation0"
    target_count = 10

    while True:
        it_counter += 1
        json_count = sum(1 for file in os.listdir(folder_path) if file.endswith('.json'))
    
        if json_count >= target_count:
            print(f"Found {json_count} JSON files, exiting the loop.")
            break

    
        population = generate_population(50, num_of_rooms)
        # Chunk population into groups of 4
        population_chunks = list(chunks(population, 10))

        for chunk_index, chunk in enumerate(population_chunks):
            print(f"Chunk {chunk_index} Size: {len(chunk)}")
            print(chunk)
            with Pool(10) as p:  # Adjust pool size as per chunk size
                try:
                    p.map(run_simulation, [(plan, chunk_index, i, it_counter) for i, plan in enumerate(chunk)])
                except IndexError as e:
                    print(f"Error on chunk {chunk_index} with size {len(chunk)}")
                    print(e)  # Print the exception for debugging
                    continue  # Continue to the next chunk

                #p.map(run_simulation, [(plan, chunk_index, i) for i, plan in enumerate(chunk)])
            

# THIS IS THE MAIN WAY OF EVOLVING A POPULATION

# PLEASE SET THE FOLDER PATH TO SAVE THE JSONS
# Please set the target_count to match the amount of floorplans in the total population of each generation
# target_count = 10
# Please set the pairs (100) to match the amount of floorplans to be picked for mutation and crossover
# pairs = [select_chromosomes(population) for _ in range(100)]
# Please change the chunks if you would like to process more in a single iteration, I usually make enough chunks to match the Pool size
# mutated_population_chunks = list(chunks(mutated_population, 10))
# In Pool(10), 10 can be swapped with any amount and that will initiate that amount of multiprocessing pools
# Pool(10)

 
if __name__ == '__main__':
    generations = 2
    num_of_rooms = 5
    for i in range(1, generations + 1):
        folder_path = f"D:\ProgrammingF#\Master's thesis\generation{i}"
        target_count = 100
        previous_gen = i-1
        population = open_generation(f"D:\\ProgrammingF#\\Master's thesis\\generation{previous_gen}")

        while True:
            json_count = sum(1 for file in os.listdir(folder_path) if file.endswith('.json'))
    
            if json_count >= target_count:
                print(f"Found {json_count} JSON files, exiting the loop.")
                break

            
            pairs = [select_chromosomes(population) for _ in range(100)]
            # Chunk population into groups of 4

            mutated_population = []
            while len(mutated_population) < target_count:
                    for pair in pairs:
                        if len(mutated_population) >= target_count:
                            break
                        # Perform mutation or crossover
                        new_child = mutate_or_crossover((pair[0], pair[1], num_of_rooms))

                        if new_child:
                            mutated_population.append(new_child)

            mutated_population_chunks = list(chunks(mutated_population, 10))
            # Create a pool of workers and run the simulations
            for chunk_index, chunk in enumerate(mutated_population_chunks):
                with Pool(10) as p:
                    try:
                        argument_list = []
                        for j, child in enumerate(chunk):
                            print(f"Processing pair {j}: {pair}")
                            argument_list.append((child, i, chunk_index, j, num_of_rooms, f"D:\\ProgrammingF#\\Master's thesis\\generation{i}\\"))
                        try:
                            p.map(run_mutated_simulation, argument_list)
                        except Exception as e:
                            print(f"Error on chunk {chunk_index} with size {len(chunk)}")
                            raise
                    except IndexError as e:
                        print(f"----------------> Error when processing params: {pair}")
                        raise







# test mutation
"""
with open("D:\\ProgrammingF#\\Master's thesis\\sample_1.json", "r") as infile:
    floorplan2 = json.load(infile)

floorplan2 = Floorplan.from_dict(floorplan2)

tree_string2 = tree_to_string(floorplan2.slicing_tree_root)
print(tree_string2)

mutated_floorplan = mutate_floorplan(floorplan2, 0.5)
mutated_tree_string = tree_to_string(mutated_floorplan.slicing_tree_root)
print(mutated_tree_string)
mutated_plan = build_slicing_tree_from_string(mutated_tree_string, ROOM_TYPES)
fp = Floorplan(num_rooms, ROOM_TYPES, mutated_plan)
"""


#mutate_or_crossover(population)

# load two jsons and mutate
"""
with open("D:\\ProgrammingF#\\Master's thesis\\sample_0.json", "r") as infile:
    floorplan1 = json.load(infile)

with open("D:\\ProgrammingF#\\Master's thesis\\sample_1.json", "r") as infile:
    floorplan2 = json.load(infile)

floorplan1 = Floorplan.from_dict(floorplan1)
floorplan2 = Floorplan.from_dict(floorplan2)

tree_string1 = tree_to_string(floorplan1.slicing_tree_root)
tree_string2 = tree_to_string(floorplan2.slicing_tree_root)
print(tree_string1)
print(tree_string2)

mutated_floorplan = mutate_floorplan(floorplan1, 0.5)
mutated_tree_string = tree_to_string(mutated_floorplan.slicing_tree_root)
print(mutated_tree_string)
mutated_plan = build_slicing_tree_from_string(mutated_tree_string, ROOM_TYPES)
fp = Floorplan(num_rooms, ROOM_TYPES, mutated_plan)
draw_floorplan(fp.rooms, screen)
pygame.display.update()


# step 2 crossover
# check matching parenthesis
crossover_point1, crossover_point2 = set_crossover_point(floorplan1.slicing_tree_root, floorplan2.slicing_tree_root)

crossover_index1 = tree_string1.find(crossover_point1.value + "{:.2f}".format(crossover_point1.cut_percentage))
print(crossover_index1)
crossover_index2 = tree_string2.find(crossover_point2.value + "{:.2f}".format(crossover_point2.cut_percentage))
print(crossover_index2)

child_tree_string1 = tree_string1[:crossover_index1] + tree_string2[crossover_index2:]
child_tree_string2 = tree_string2[:crossover_index2] + tree_string1[crossover_index1:]

child1 = build_slicing_tree_from_string(child_tree_string1, ROOM_TYPES)
child2 = build_slicing_tree_from_string(child_tree_string2, ROOM_TYPES)


fp = Floorplan(num_rooms, ROOM_TYPES, child2)

draw_floorplan(fp.rooms, screen)
pygame.display.update()
print("Performed crossover between two chromosomes")
"""

