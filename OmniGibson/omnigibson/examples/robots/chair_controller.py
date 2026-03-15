# import torch as th
# import math


# class ChairController:

#     def __init__(self, env, robot):
#         self.env = env
#         self.robot = robot
#         self.chair = self._find_chair()

#     def _find_chair(self):
#         for obj in self.env.scene.objects:
#             if "chair" in obj.name.lower():
#                 return obj
#         raise RuntimeError("Chair not found")

#     def get_action(self):

#         robot_pos, robot_orn = self.robot.get_position_orientation()
#         chair_pos, _ = self.chair.get_position_orientation()

#         dx = chair_pos[0] - robot_pos[0]
#         dy = chair_pos[1] - robot_pos[1]

#         dist = (dx**2 + dy**2) ** 0.5

#         yaw = math.atan2(
#             2 * (robot_orn[3] * robot_orn[2] + robot_orn[0] * robot_orn[1]),
#             1 - 2 * (robot_orn[1] ** 2 + robot_orn[2] ** 2),
#         )

#         target_yaw = math.atan2(dy, dx)
#         error = target_yaw - yaw
#         error = (error + math.pi) % (2 * math.pi) - math.pi

#         forward = 0.4
#         turn = 2.0 * error

#         if dist < 0.5:
#             forward = 0.0

#         return th.tensor([forward, turn, 0.0, 0.0])


import torch as th
import math


class ChairController:

    def __init__(self, env, robot):
        self.env = env
        self.robot = robot
        self.chair = self._find_chair()
        self.push_steps = 0

    def _find_chair(self):
        for obj in self.env.scene.objects:
            if "chair" in obj.name.lower():
                return obj
        raise RuntimeError("Chair not found")

    def get_action(self):

        robot_pos, robot_orn = self.robot.get_position_orientation()
        chair_pos, _ = self.chair.get_position_orientation()

        dx = chair_pos[0] - robot_pos[0]
        dy = chair_pos[1] - robot_pos[1]

        dist = math.sqrt(dx*dx + dy*dy)

        # robot yaw
        yaw = math.atan2(
            2*(robot_orn[3]*robot_orn[2] + robot_orn[0]*robot_orn[1]),
            1 - 2*(robot_orn[1]**2 + robot_orn[2]**2)
        )

        target_yaw = math.atan2(dy, dx)
        err = target_yaw - yaw
        err = (err + math.pi) % (2*math.pi) - math.pi

        turn = 2.5 * err
        

        # approach phase
        if dist > 1.5:
            forward = 0.6

        # attack phase
        elif dist > 0.6:
            forward = 1.8

        # contact phase (push)
        else:
            forward = 2.0
            self.push_steps += 1

        # stop pushing after some time
        # if self.push_steps > 40:
        #     forward = 0

        return th.tensor([forward, turn, 0.0, 0.0])