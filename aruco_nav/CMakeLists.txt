cmake_minimum_required(VERSION 3.8)
project(aruco_nav)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/Search.action"
  DEPENDENCIES geometry_msgs
)

install(PROGRAMS scripts/search.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME search
)
install(FILES scripts/planner.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
