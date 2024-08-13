import os
import gtsam
from typing import List
import argparse

def rm_robots_g2o(
    g2o_lines: List[str], robot_ids: List[int] = None, robot_letters: List[str] = None
) -> List[str]:
    """
    Removes robot vertices and edges from a g2o file.

    Args:
        g2o_lines (List[str]): Lines of original g2o file.
        robot_ids (List[int], optional): Numerical robot indices. Defaults to None.
        robot_letters (List[str], optional): Gtsam robot letters. Defaults to None.

    Returns:
        List[str]: Lines of g2o file with robot vertices and edges removed.
    """
    if robot_ids is None and robot_letters is None:
        raise ValueError("Either robot_ids or robot_letters must be provided.")

    if robot_ids is not None and robot_letters is not None:
        raise ValueError("Only one of robot_ids or robot_letters can be provided.")

    if robot_ids is not None:
        robot_letters = [chr(ord("a") + robot_id) for robot_id in robot_ids]
    
    print(f"Removing robots: {robot_letters}")

    new_g2o_lines = []
    for line in g2o_lines:
        data = line.strip().split()
        if data[0] == "VERTEX_SE3:QUAT":
            line_robot_letter = chr(gtsam.Symbol(int(data[1])).chr())
            if line_robot_letter in robot_letters:
                continue
        elif data[0] == "EDGE_SE3:QUAT":
            line_robot_letters = [chr(gtsam.Symbol(int(data[i])).chr()) for i in range(1, 3)]
            if any([robot_letter in line_robot_letters for robot_letter in robot_letters]):
                continue
        new_g2o_lines.append(line)

    return new_g2o_lines

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Remove robots from g2o file')
    parser.add_argument('input', type=str, help='Path to g2o file')
    parser.add_argument('output', type=str, help='Path to output g2o')
    parser.add_argument('-n', '--robot_ids', type=int, nargs='+', help='Robot indices to remove')
    parser.add_argument('-l', '--robot_letters', type=str, nargs='+', help='Robot letters to remove')
    args = parser.parse_args()
    
    with open(args.input, "r") as f:
        g2o_lines = f.readlines()
        f.close()
    
    new_g2o_lines = rm_robots_g2o(g2o_lines, args.robot_ids, args.robot_letters)
    
    with open(args.output, "w") as f:
        f.writelines(new_g2o_lines)
        f.close()