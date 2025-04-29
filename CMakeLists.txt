cmake_minimum_required(VERSION 3.8)
project(dialog_example)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  dialog_example/dialog_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY dialog_example/
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.jsonc"
)

ament_package()
