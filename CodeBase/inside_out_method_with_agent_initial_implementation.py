import pygame
import random
import sys
from collections import namedtuple
import math

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

pygame.init()
clock = pygame.time.Clock()
pygame.font.init()
room_number_font = pygame.font.Font(None, 36)
object_font = pygame.font.Font(None, 13)
font = pygame.font.SysFont('Arial', 16)

GRID_SIZE = 10

# Define room types with their size ranges

RoomType = namedtuple('RoomType', ['name', 'min_size', 'max_size'])
ROOM_TYPES = [
    RoomType('bedroom', (100, 100), (200, 200)),
    RoomType('living_room', (200, 200), (400, 300)),
    RoomType('kitchen', (50, 50), (200, 200)),
    RoomType('bathroom', (50, 50), (150, 150))
]

OBJECTS_BY_ROOM_TYPE = {
    'bedroom': ['bed', 'wardrobe', 'dresser'],
    'living_room': ['couch', 'coffee_table', 'tv'],
    'kitchen': ['refrigerator', 'oven', 'sink', 'table', 'chairs'],
    'bathroom': ['sink', 'toilet', 'shower']
}


# generates a random room size within the specified range for the given room type.
def random_room_size(room_type):
    width = random.randint(room_type.min_size[0] // GRID_SIZE, room_type.max_size[0] // GRID_SIZE) * GRID_SIZE
    length = random.randint(room_type.min_size[1] // GRID_SIZE, room_type.max_size[1] // GRID_SIZE) * GRID_SIZE
    return width, length


class Object:
    def __init__(self, name, position):
        self.name = name
        self.position = position
    
    def draw(self, surface):
        # Draw the object at its position
        pygame.draw.circle(surface, (0, 255, 0), self.position, 10)

        inf = object_font.render(self.name+" " +str(self.position) , True, (0, 0, 0))
        screen.blit(inf, self.position)


# represents a room, has attributes for the room's rectangular dimensions, color, room type, and walls (a list of Wall instances)
class Room:
    def __init__(self, x, y, width, length, room_type):
        self.rect = pygame.Rect(x, y, width, length)
        self.square_meters = width * length
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.room_type = room_type
        self.objects = []  # List of objects in the room
        self.number = None
        self.connection_points = []
        self.points = []

        wall_thickness = 4
        half_thickness = wall_thickness // 2
        self.walls = [
            Wall(x - half_thickness, y - wall_thickness, width + wall_thickness, wall_thickness,  "horizontal", "top"),  # top
            Wall(x - half_thickness, y + length, width + wall_thickness, wall_thickness,  "horizontal", "bottom"),  # bottom
            Wall(x - wall_thickness, y - half_thickness, wall_thickness, length + wall_thickness,  "vertical", "left"),  # left
            Wall(x + width, y - half_thickness, wall_thickness, length + wall_thickness,  "vertical", "right"),  # right
        ]
        # Add objects to the room
        for object_name in OBJECTS_BY_ROOM_TYPE[self.room_type.name]:
            # Choose a random position for the object within the room
            x = random.randint(self.rect.x, self.rect.x + self.rect.width)
            y = random.randint(self.rect.y, self.rect.y + self.rect.height)
            random_spot = x,y
            new_object = Object(object_name, random_spot)
            self.objects.append(new_object)

    def draw(self, screen):
        # Draw the room
        pygame.draw.rect(screen, self.color, self.rect)

        # Draw the walls
        for wall in self.walls:
            pygame.draw.rect(screen, (170, 170, 170), wall.rect)

        # Draw objects in the room
        for obj in self.objects:
            obj.draw(screen)

        for connection_point in self.connection_points:
            pygame.draw.rect(screen, (255, 0, 0), (connection_point[0], connection_point[1], GRID_SIZE, GRID_SIZE))


# represents a wall of a room, has attributes for the wall's rectangular dimensions 
# orientation (horizontal or vertical), position (top, bottom, left, or right), and a boolean flag active

class Wall:
    def __init__(self, x, y, width, height,  orientation, position):
        self.rect = pygame.Rect(x, y, width, height)
        self.orientation = orientation
        self.position = position
        self.active = False  # Door is inactive by default




# tries to place a new room adjacent to an existing room
# iterates through all possible adjacent directions (top, bottom, left, or right) and checks if there's no overlapping with other rooms
def place_room_adjacent(existing_room, rooms):
    adjacent_directions = ["top", "bottom", "left", "right"]
    random.shuffle(adjacent_directions)

    best_new_room = None
    best_new_room_min_distance = float('inf')

    for direction in adjacent_directions:
        room_type = random.choice(ROOM_TYPES)
        width, length = random_room_size(room_type)
        start = end = None

        if direction == "top":
            start = existing_room.rect.x
            end = existing_room.rect.x + existing_room.rect.width - width
            y = existing_room.rect.y - length
        elif direction == "bottom":
            start = existing_room.rect.x
            end = existing_room.rect.x + existing_room.rect.width - width
            y = existing_room.rect.y + existing_room.rect.height
        elif direction == "left":
            start = existing_room.rect.y
            end = existing_room.rect.y + existing_room.rect.height - length
            x = existing_room.rect.x - width
        elif direction == "right":
            start = existing_room.rect.y
            end = existing_room.rect.y + existing_room.rect.height - length
            x = existing_room.rect.x + existing_room.rect.width

        # Snap the start and end points to the grid
        start = (start // GRID_SIZE) * GRID_SIZE
        end = (end // GRID_SIZE) * GRID_SIZE

        # Iterate through the snapped grid points and try placing the new room at each position
        for pos in range(start, end + 1, GRID_SIZE):
            new_room = None
            if direction in ["top", "bottom"]:
                x = pos
            else:
                y = pos

            new_room = Room(x, y, width, length, room_type)

            # Check for overlapping
            overlapping = False
            min_distance = float('inf')
            for room in rooms:
                if new_room.rect.colliderect(room.rect):
                    overlapping = True
                    break
                else:
                    distance = math.sqrt((room.rect.x - new_room.rect.x)**2 + (room.rect.y - new_room.rect.y)**2)
                    min_distance = min(min_distance, distance)

            if not overlapping and min_distance < best_new_room_min_distance:
                best_new_room = new_room
                best_new_room_min_distance = min_distance

    return best_new_room






#  tries to place a new room adjacent to any of the existing rooms. It iterates through the rooms and calls place_room_adjacent for each room
def place_room_adjacent_to_any(rooms):
    random.shuffle(rooms)
    for existing_room in rooms:
        new_room = place_room_adjacent(existing_room, rooms)
        if new_room:
            return new_room
    return None




def draw_grid(screen, grid_size, color):
    screen_width, screen_height = screen.get_size()
    
    for x in range(0, screen_width, grid_size):
        pygame.draw.line(screen, color, (x, 0), (x, screen_height))
    
    for y in range(0, screen_height, grid_size):
        pygame.draw.line(screen, color, (0, y), (screen_width, y))




def is_adjacent(room1, room2, distance=GRID_SIZE):
    min_overlap = 3 * GRID_SIZE

    # Check for adjacency horizontally
    if (room1.rect.y >= room2.rect.y - room1.rect.height - distance and
            room1.rect.y <= room2.rect.y + room2.rect.height + distance):
        overlap_horizontal = min(room1.rect.y + room1.rect.height, room2.rect.y + room2.rect.height) - max(room1.rect.y, room2.rect.y)
        if (abs(room1.rect.x - room2.rect.x - room2.rect.width) <= distance or abs(room2.rect.x - room1.rect.x - room1.rect.width) <= distance) and overlap_horizontal >= min_overlap:
            
            connection_y = room1.rect.y + (overlap_horizontal // 2)
            connection_x = room1.rect.x + room1.rect.width if room1.rect.x < room2.rect.x else room2.rect.x + room2.rect.width
            return True, (connection_x, connection_y)

    # Check for adjacency vertically
    if (room1.rect.x >= room2.rect.x - room1.rect.width - distance and
            room1.rect.x <= room2.rect.x + room2.rect.width + distance):
        overlap_vertical = min(room1.rect.x + room1.rect.width, room2.rect.x + room2.rect.width) - max(room1.rect.x, room2.rect.x)
        if (abs(room1.rect.y - room2.rect.y - room2.rect.height) <= distance or abs(room2.rect.y - room1.rect.y - room1.rect.height) <= distance) and overlap_vertical >= min_overlap:
            
            connection_x = room1.rect.x + (overlap_vertical // 2)
            connection_y = room1.rect.y + room1.rect.height if room1.rect.y < room2.rect.y else room2.rect.y + room2.rect.height
            return True, (connection_x, connection_y)

    return False, None



def draw_extensions(screen, rooms, distance=GRID_SIZE):
    for room in rooms:
        extensions = [
            pygame.Rect(room.rect.x, room.rect.y - distance, room.rect.width, distance),  # Top
            pygame.Rect(room.rect.x, room.rect.y + room.rect.height, room.rect.width, distance),  # Bottom
            pygame.Rect(room.rect.x - distance, room.rect.y, distance, room.rect.height),  # Left
            pygame.Rect(room.rect.x + room.rect.width, room.rect.y, distance, room.rect.height)  # Right
        ]

        for extension in extensions:
            pygame.draw.rect(screen, (0, 0, 255), extension, 1)  # Draw the extension with a blue outline


def create_floor_plan_graph(rooms):
    graph = {room: [] for room in rooms}  # Initialize the graph with empty adjacency lists for each room

    for i, room1 in enumerate(rooms):
        for room2 in rooms[i + 1:]:
            is_adj, connection_point = is_adjacent(room1, room2)
            room1.points.append(connection_point)
            room2.points.append(connection_point)
            if is_adj:
                # If rooms are adjacent, add an edge in the graph (undirected)
                graph[room1].append(room2)
                graph[room2].append(room1)

    return graph




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
        labels = {room: f"{room.room_type.name} {room.number}" for room in G.nodes()}


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
    

class Agent:
    def __init__(self, start_room, all_rooms):
        self.position = start_room.rect.center
        self.destination = self.position
        self.current_room = start_room
        self.all_rooms = all_rooms
        self.interaction_duration = 0
        self.path = []
        self.previous_target_room = None
        self.target_room = None

        # Create a floor_plan_graph
        self.floor_plan_graph = self.create_floor_plan_graph(all_rooms)

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 0, 255), self.position, 10)
        task_text = font.render(" " + str(self.interaction_duration)+ " ", True, (255,255,255))
        screen.blit(task_text, self.position)

    def create_floor_plan_graph(self, rooms):
        # Create the graph with rooms as nodes and connection_points as edges
        graph = {}
        for room in rooms:
            graph[room] = set()
            for connection_point in room.connection_points:
                connected_room = self.get_room_at_point(connection_point)
                if connected_room is not None and connected_room != room:
                    graph[room].add(connected_room)
        return graph

    def get_room_at_point(self, point):
        for room in self.all_rooms:
            if room.rect.collidepoint(point):
                return room
        return None

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

    def update(self, dt):
        if self.interaction_duration > 0:
            self.interaction_duration -= dt
            return

        if self.position == self.destination:
            self.previous_target_room = self.target_room
            # Choose a new destination
            self.target_room = random.choice(self.all_rooms)
            target_object = random.choice(self.target_room.objects)
            self.destination = target_object.position

            # Update the path of rooms
            self.current_room = self.get_room_at_point(self.position)
            if self.current_room is None:
                self.current_room = self.previous_target_room
            room_path = self.bfs_shortest_path(self.current_room, self.target_room)
            self.path = []

            # Update the path of connection points
            for i in range(len(room_path) - 1):
                current_room = room_path[i]
                next_room = room_path[i + 1]

                for connection_point in current_room.connection_points:
                    if self.get_room_at_point(connection_point) == next_room:
                        self.path.append(connection_point)
                        break

            self.path.append(self.destination)
            self.visited = set()  # Reset the visited set

        else:
            # Move towards the next point in the path
            if len(self.path) > 0:
                direction = (self.path[0][0] - self.position[0], self.path[0][1] - self.position[1])
                distance = math.sqrt(direction[0] ** 2 + direction[1] ** 2)

                if distance < 1:
                    self.position = self.path[0]
                    self.path.pop(0)  # Remove the reached point from the path
                else:
                    speed = 50  # pixels per second
                    move_vector = (direction[0] / distance * speed * dt, direction[1] / distance * speed * dt)
                    new_position = (self.position[0] + move_vector[0], self.position[1] + move_vector[1])
                    self.position = new_position

            # fully connected graph, points to the door node, the door node points to the other room
            # list of destinations for other room objects
            # list of destinations is generated by the a* walking through that graph through the doors
            # pick target room 
            # traverse with a*, build  a list of doors as destinations
            # move to door, if on door, remove door from list, continue to next door
            # repeat until on the desired room rect, move to object

            # treat objects and doors as the same thing

# Define screen and room dimensions
screen_width = 1100
screen_height = 1100
room_width = random.randint(100, 200)
room_length = random.randint(100, 200)
extension = 10
num_rooms = 5

# Initialize screen and set room placement flag
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Floor Plan Generator')
rooms = []

# Place the first room in the middle of the screen
x = (screen_width - room_width) // 2
y = (screen_height - room_length) // 2
first_room = Room(x, y, room_width, room_length,  random.choice(ROOM_TYPES))
rooms.append(first_room)

# Place additional rooms adjacent to existing rooms
while len(rooms) < num_rooms:
    new_room = place_room_adjacent_to_any(rooms)
    if new_room:
        rooms.append(new_room)

for i, room in enumerate(rooms, start=1):
    room.number = i

floor_plan_graph = create_floor_plan_graph(rooms)

graph_visualization = GraphVisualization.from_floor_plan_graph(floor_plan_graph)
graph_visualization.visualize()

for i, room1 in enumerate(rooms):
    for j, room2 in enumerate(rooms):
        if i != j:
            is_adj, connection_point = is_adjacent(room1, room2)
            if is_adj:
                if connection_point not in room1.connection_points:
                    room1.connection_points.append(connection_point)
                if connection_point not in room2.connection_points:
                    room2.connection_points.append(connection_point)


# connection points are not always belonging to rooms

start_room = random.choice(rooms)
agent = Agent(start_room, rooms)


for i, room in enumerate(rooms):
    # Draw the room
    print(room.room_type.name)
    print(i)
    print(room.connection_points)


while True:
    #print(agent.path)
    #print(agent.current_room.room_type.name)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    dt = clock.tick(60)  # Add this line at the beginning of your main loop to get delta time in milliseconds

    screen.fill((255, 255, 255))  # Fill the screen with a white background

    # Draw the grid
    draw_grid(screen, GRID_SIZE, (200, 200, 200))

    # Draw the extensions for checking adjacency
    draw_extensions(screen, rooms)

    # Draw all the rooms with their walls, and room numbers
    for room in rooms:
        # Draw the room
        room.draw(screen)

    # Draw room numbers
    for i, room in enumerate(rooms, start=1):
        room_number = room_number_font.render(str(i) + " " + room.room_type.name, True, (0, 0, 0))
        room_number_rect = room_number.get_rect()
        room_number_rect.center = room.rect.center
        screen.blit(room_number, room_number_rect)

    # Update and draw the agent
    agent.update(dt / 1000.0)  # Convert dt to seconds
    agent.draw(screen)

    pygame.display.flip()




