#!/bin/bash

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building filc package ===${NC}"

# 워크스페이스로 이동
cd /home/user1/ROS2_Workspace/ros2_ws

# 빌드
echo -e "${YELLOW}Building with colcon...${NC}"
colcon build --packages-select filc --symlink-install

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# 환경 설정
echo -e "${YELLOW}Sourcing workspace...${NC}"
source install/setup.bash

echo -e "${GREEN}=== Available nodes ===${NC}"
ros2 pkg executables filc

echo -e "${GREEN}=== Build complete! ===${NC}"
echo ""
echo "To run the test interpolation node:"
echo "  ros2 run filc test_interpolation_node"
echo ""
echo "To monitor the output:"
echo "  python3 src/filc/scripts/visualize_interpolation.py"
echo ""
echo "To check topics:"
echo "  ros2 topic list"
echo "  ros2 topic echo /ouster/interpolated_points --no-arr"