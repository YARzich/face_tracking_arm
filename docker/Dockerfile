# syntax=docker/dockerfile:1
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

FROM ros:jazzy-ros-base-noble AS build
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG XARM_REVISION=3dc2b5e8294758d96b54b15fa5920d581b7cbb3d
ARG CAMERACTRLS_REVISION=6f388257ac21a0e91b143ad11cb2457036fa2c27
ARG XARM_PYTHON_REVISION=d911319cebc45142613e18086aabb8067b18ab7f
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt/face_tracking_ws
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl ninja-build python3-venv && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none --sparse --branch jazzy \
      https://github.com/xArm-Developer/xarm_ros2.git src/xarm_ros2 && \
    git -C src/xarm_ros2 sparse-checkout set \
      xarm_sdk xarm_msgs xarm_description xarm_api xarm_controller && \
    git -C src/xarm_ros2 checkout "$XARM_REVISION" && \
    git -C src/xarm_ros2 submodule update --init --recursive xarm_sdk/cxx
COPY docker/patches/xarm-readiness.patch /tmp/xarm-readiness.patch
COPY docker/patches/hardware_activation_guard.hpp /opt/face_tracking_ws/src/xarm_ros2/xarm_controller/include/xarm_controller/hardware_activation_guard.hpp
RUN git -C src/xarm_ros2 apply --check /tmp/xarm-readiness.patch && \
    git -C src/xarm_ros2 apply /tmp/xarm-readiness.patch
# Keep the LGPL tool separate from the MIT application, with its complete source.
RUN git init /opt/cameractrls && \
    git -C /opt/cameractrls remote add origin https://github.com/soyersoyer/cameractrls.git && \
    git -C /opt/cameractrls fetch --depth 1 origin "$CAMERACTRLS_REVISION" && \
    git -C /opt/cameractrls checkout --detach FETCH_HEAD && \
    git -C /opt/cameractrls archive HEAD -o /opt/cameractrls-source.tar
COPY package.xml /opt/manifests/face_tracking_arm/package.xml
RUN for package in xarm_sdk xarm_msgs xarm_description xarm_api xarm_controller; do \
      mkdir -p "/opt/manifests/$package"; \
      cp "src/xarm_ros2/$package/package.xml" "/opt/manifests/$package/"; \
    done
RUN apt-get update && rosdep update --rosdistro jazzy && \
    rosdep install --from-paths /opt/manifests --ignore-src -y --rosdistro jazzy \
      -t build -t buildtool -t build_export -t exec \
      --skip-keys 'gz_ros2_control ros_gz_bridge ros_gz_sim rviz2' && \
    rm -rf /var/lib/apt/lists/*
COPY config/perception_requirements.txt /opt/perception_requirements.txt
RUN python3 -m venv --system-site-packages /opt/face_tracking_venv && \
    /opt/face_tracking_venv/bin/python -m pip install --no-cache-dir \
      -r /opt/perception_requirements.txt && \
    /opt/face_tracking_venv/bin/python -m pip install --no-cache-dir --no-deps \
      --target /opt/face_tracking_sdk \
      "https://github.com/xArm-Developer/xArm-Python-SDK/archive/${XARM_PYTHON_REVISION}.tar.gz" && \
    curl -fL --retry 3 \
      "https://raw.githubusercontent.com/xArm-Developer/xArm-Python-SDK/${XARM_PYTHON_REVISION}/LICENSE" \
      -o /opt/face_tracking_sdk/LICENSE.xarm-python-sdk && \
    mkdir -p /opt/face_tracking_models && \
    curl -fL --retry 3 \
      https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
      -o /opt/face_tracking_models/face_detection_yunet_2023mar.onnx && \
    echo '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  /opt/face_tracking_models/face_detection_yunet_2023mar.onnx' | sha256sum -c -
COPY CMakeLists.txt package.xml LICENSE THIRD_PARTY_NOTICES.md src/face_tracking_arm/
COPY config src/face_tracking_arm/config
COPY description src/face_tracking_arm/description
COPY include src/face_tracking_arm/include
COPY msg src/face_tracking_arm/msg
COPY src src/face_tracking_arm/src
COPY python src/face_tracking_arm/python
COPY scripts src/face_tracking_arm/scripts
COPY launch src/face_tracking_arm/launch
COPY worlds src/face_tracking_arm/worlds
RUN source /opt/ros/jazzy/setup.bash && \
    CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --merge-install --executor sequential \
      --packages-select xarm_sdk xarm_msgs xarm_description xarm_api xarm_controller face_tracking_arm \
      --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja -DBUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE=Release
FROM ros:jazzy-ros-base-noble AS runtime
ARG FACE_TRACKING_SOURCE_SHA256=unverified
LABEL org.opencontainers.image.title="face_tracking_arm" \
      io.face_tracking_arm.launch-schema="2" \
      io.face_tracking_arm.source-sha256="${FACE_TRACKING_SOURCE_SHA256}"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
COPY --from=build /opt/manifests /opt/manifests
RUN apt-get update && rosdep update --rosdistro jazzy && \
    rosdep install --from-paths /opt/manifests --ignore-src -y --rosdistro jazzy \
      --dependency-types exec --skip-keys 'gz_ros2_control ros_gz_bridge ros_gz_sim rviz2' && \
    apt-get install -y --no-install-recommends \
      ros-jazzy-camera-calibration v4l-utils libgomp1 && \
    rm -rf /var/lib/apt/lists/* /opt/manifests
COPY --from=build /opt/face_tracking_ws/install /opt/face_tracking_ws/install
COPY --from=build /opt/face_tracking_venv /opt/face_tracking_venv
COPY --from=build /opt/face_tracking_models /opt/face_tracking_models
COPY --from=build /opt/face_tracking_sdk /opt/face_tracking_sdk
COPY --from=build /opt/cameractrls/cameractrls.py /opt/cameractrls/cameractrls.py
COPY --from=build /opt/cameractrls/LICENSE /opt/cameractrls/LICENSE
COPY --from=build /opt/cameractrls-source.tar /usr/share/doc/face_tracking_arm/third_party/cameractrls-source.tar
RUN cp /usr/share/common-licenses/LGPL-3 /opt/cameractrls/COPYING.LESSER && \
    cp /usr/share/common-licenses/GPL-3 /opt/cameractrls/COPYING
COPY --from=build /opt/face_tracking_ws/src/xarm_ros2/LICENSE /usr/share/doc/face_tracking_arm/third_party/xarm_ros2-BSD.txt
COPY --from=build /opt/face_tracking_ws/src/xarm_ros2/xarm_sdk/cxx/LICENSE /usr/share/doc/face_tracking_arm/third_party/xarm_sdk-BSD.txt
COPY docker/licenses /usr/share/doc/face_tracking_arm/third_party
COPY docker/patches /usr/share/doc/face_tracking_arm/third_party/xarm-local-patches
COPY docker/entrypoint.sh docker/runtime.py docker/fastdds.xml /opt/face_tracking_container/
ENV HOME=/tmp/face_tracking_home \
    PYTHONPATH=/opt/face_tracking_sdk \
    ROS_DOMAIN_ID=71 \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTRTPS_DEFAULT_PROFILES_FILE=/opt/face_tracking_container/fastdds.xml
WORKDIR /tmp
STOPSIGNAL SIGINT
ENTRYPOINT ["/bin/bash", "/opt/face_tracking_container/entrypoint.sh"]
CMD ["help"]
