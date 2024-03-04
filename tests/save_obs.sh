# 9:45 - 10:45
python3 save_observations.py \
    -b /home/masonbp/data/west_point_2023/1117_04_loc_tower_loop_far_side_loop_o3_CCW/align.bag \
    --pose-topic "/acl_jackal2/kimera_vio_ros/odometry" --camera-topic "/acl_jackal2/forward/color" \
    --t0 585 --tf 645 -o /home/masonbp/results/west_point_2023/segment_observations/1117_04_585_645.pkl

# python3 save_observations.py \
#     -b /home/masonbp/data/west_point_2023/1117_02_map_tower_third_origin/dataset.bag \
#     --pose-topic "/acl_jackal2/kimera_vio_ros/odometry" --camera-topic "/acl_jackal2/forward/color" \
#     --t0 165 --tf 225 -o /home/masonbp/results/west_point_2023/segment_observations/1117_02_165_225.pkl