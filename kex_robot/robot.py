import numpy as np
import random


def l1norm_dist(p1: [], p2: []) -> float:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class Robot():
    NOTHING = 0
    WALKING = 1
    DROPPING = 2

    def __init__(self, robot: "RoboticWareHouseRobot", gym: "RoboticWareHouse",
                 capacity: int, reservations: set, swarm: "[Robot]"):
        self.robot = robot
        self.gym = gym
        self.capacity = capacity
        self.reservations = reservations
        self.swarm = swarm

        self.state = Robot.NOTHING
        self.directions = [gym.DOWN, gym.LEFT, gym.UP, gym.RIGHT]
        self.perpendicular = [1, 2, 1, 2]
        self.anti = [2, 3, 0, 1]

        self.target = None
        self.prev_movement = None

    def walkable(self, p: []) -> bool:
        return self.gym.in_map(
            p[0], p[1]) and self.gym.map[p[0]][p[1]][0] == self.gym.TILE_ID

    def walkable_near(self, po: []) -> []:
        if self.walkable(po):
            return po

        for d in self.directions:
            p = [d[0] + po[0], d[1] + po[1]]
            if self.walkable(p):
                return p

    def can_pickup(self, position, packages):
        for p in packages:
            if l1norm_dist(position, p.start) == 1 and p in self.robot.reservations:
                return True
        return False

    def can_drop(self, position, packages):
        for p in packages:
            if l1norm_dist(position, p.dropoff) == 1:
                return True
        return False

    def get_free(self, packages):
        free = []
        for package in packages:
            if package not in self.reservations:
                free.append(package)
        return free

    def closest_package(self, positon, packages):
        return packages[np.argmin(
            list(
                map(lambda package: l1norm_dist(positon, package.start),
                    packages)))]

    def closest_dropoff(self, position, packages):
        return packages[np.argmin(
            list(
                map(lambda package: l1norm_dist(position, package.dropoff),
                    packages)))].dropoff

    def move(self):
        if not self.target:
            return None

        if list(self.robot.position) == list(self.target):
            return None

        direction = [
            self.target[0] - self.robot.position[0],
            self.target[1] - self.robot.position[1]
        ]
        alternative = [direction[0], direction[1]]
        """ We cannot walk diagonally so need to choose a direction. """
        if not (direction[0] == 0 or direction[1] == 0):
            direction[np.argmin(np.abs(direction))] = 0
            alternative[np.argmax(np.abs(direction))] = 0

        direction = [np.sign(direction[0]), np.sign(direction[1])]

        alternative = [np.sign(alternative[0]), np.sign(alternative[1])]

        perp = [direction[1], direction[0]]

        position = [
            self.robot.position[0] + direction[0],
            self.robot.position[1] + direction[1]
        ]

        alt_position = [
            self.robot.position[0] + alternative[0],
            self.robot.position[1] + alternative[1]
        ]

        perp_position = [
            self.robot.position[0] + perp[0], self.robot.position[1] + perp[1]
        ]

        movement = self.directions.index(direction)
        if self.walkable(
                position
        ) and position != self.robot.position and self.anti[movement] != self.prev_movement:
            self.prev_movement = movement
            return self.prev_movement
        elif self.walkable(alt_position) and alternative != [
                0, 0
        ] and self.anti[self.directions.index(
                alternative)] != self.prev_movement:
            self.prev_movement = self.directions.index(alternative)
            return self.prev_movement
        elif self.walkable(perp_position) and perp != [
                0, 0
        ] and self.anti[self.directions.index(perp)] != self.prev_movement:
            self.prev_movement = self.directions.index(perp)
            return self.prev_movement

        if self.prev_movement != None:
            y, x = self.directions[self.prev_movement]
            if self.walkable(
                [self.robot.position[0] + y, self.robot.position[1] + x]):
                return self.prev_movement

        print("This will never be called.. probably bad design")
        raise Exception("Error occured")

    def dropoff_condition(self, free):
        return len(self.robot.packages) == self.capacity or\
            (self.state == Robot.DROPPING and len(self.robot.packages) >= 1) or\
            (len(free) == 0 and len(self.robot.packages) >= 1)

    def pickup_condition(self, packages):
        return len(self.robot.packages) != self.capacity and self.can_pickup(
            self.robot.position, packages)

    def should_pickup(self, free):
        package_ranking = list(
            map(lambda p: (l1norm_dist(p.start, self.robot.position), p),
                free))
        package_ranking.sort(key=lambda x: x[0])
        for dist, p in package_ranking:
            should = True
            for r in self.swarm:
                if r.state != Robot.NOTHING:
                    continue

                if l1norm_dist(r.robot.position, p.start) + 1 < dist:
                    should = False
                    break

            if should:
                return p
        return None

    def __call__(self,
                 on_map: "packages",
                 idle_task: "F: self -> []" = lambda _: [0, 0]):

        free = self.get_free(on_map)

        if self.target != None:
            movement = self.move()
            if movement != None:
                return movement
        """ If we can pickup a package, pick it up (Robots might steal eachothers packages)"""
        if self.pickup_condition(on_map):
            self.state = Robot.NOTHING
            self.prev_movement = None
            return self.gym.PICKUP_INSTRUCTION
        """ If we can drop a package we drop it and then move to drop next packages"""
        if self.dropoff_condition(free):
            self.state = Robot.DROPPING

            if self.can_drop(self.robot.position, self.robot.packages):
                self.prev_movement = None
                return self.gym.DROP_INSTRUCTION

            self.target = self.walkable_near(
                self.closest_dropoff(self.robot.position, self.robot.packages))
            movement = self.move()
            if movement != None:
                return movement
            else:
                return self.gym.DROP_INSTRUCTION
            return self.move()
        """ Get free packages """
        if free:
            go_pickup = self.should_pickup(free)
            if go_pickup:
                self.state = Robot.WALKING
                self.target = self.walkable_near(go_pickup.start)
                self.reservations.add(go_pickup)
                self.robot.reservations.add(go_pickup)
                movement = self.move()
                if movement != None:
                    return movement
                else:
                    return self.gym.PICKUP_INSTRUCTION
        """ 
        For some reasons robots forget to go for their reservations.. 
        this is a bug and this if here is a ugly fix. 
        """
        if len(self.robot.reservations) != 0:
            self.state = Robot.WALKING
            self.target = self.walkable_near(
                tuple(self.robot.reservations)[0].start)
            self.move()
        """ If there is still nothing to do, we are idle!"""
        self.state = Robot.NOTHING
        self.target = self.walkable_near(idle_task(self))

        movement = self.move()
        if movement != None:
            return movement

        return self.gym.PICKUP_INSTRUCTION
