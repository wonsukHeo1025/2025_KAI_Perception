# ConeSTELLATION Documentation

## Overview

ConeSTELLATION is a cone-based Graph SLAM system for Formula Student autonomous racing. This directory contains the core documentation for understanding, developing, and maintaining the system.

## Documentation Structure

### 📋 Core Documents
- **[PRD.md](PRD.md)** - Product Requirements Document: Vision, requirements, and success metrics
- **[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)** - Comprehensive development plan with architecture, implementation details, and roadmap

### 🔧 Technical Documentation
- **[api-reference.md](api-reference.md)** - Comprehensive API documentation for key classes and components
- **[sensor-simulation.md](sensor-simulation.md)** - IMU and GPS sensor simulation with realistic noise models
- **[ekf-fusion.md](ekf-fusion.md)** - EKF implementation details for 100Hz IMU-GPS fusion
- **[testing-framework.md](testing-framework.md)** - Motion profiles, ground truth system, and evaluation tools
- **[critical-issues.md](critical-issues.md)** - Known bugs, technical debt, and improvement roadmap

### 📊 Configuration References
- **[input_topic_form.md](input_topic_form.md)** - Actual sensor data formats from ROS2 topics
- **[topic_structure.md](topic_structure.md)** - ROS2 topics and TF tree structure

### 🐛 Development History
- **[debug_log.md](debug_log.md)** - Chronological archive of issues, solutions, and learnings (append-only)

## Quick Navigation

**New to the project?** → Start with [PRD.md](PRD.md) to understand goals and requirements

**Want to contribute?** → Read [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for current status and roadmap

**Understanding the code?** → Check [api-reference.md](api-reference.md) for class documentation

**Testing the system?** → See [testing-framework.md](testing-framework.md) and [sensor-simulation.md](sensor-simulation.md)

**Setting up sensors?** → Review [ekf-fusion.md](ekf-fusion.md) and [topic_structure.md](topic_structure.md)

**Debugging an issue?** → Search [debug_log.md](debug_log.md) and check [critical-issues.md](critical-issues.md)

## Document Maintenance Guidelines

1. **PRD.md** - Update when requirements or vision changes
2. **DEVELOPMENT_PLAN.md** - Keep current with:
   - Implementation status (move items from ❌ to 🚧 to ✅)
   - Architecture updates
   - New technical decisions
3. **API Documentation** - Update when:
   - New classes or methods are added
   - Interfaces change
   - Implementation details significantly change
4. **Technical Documentation** - Maintain accuracy of:
   - Sensor models and parameters
   - Configuration examples
   - Test procedures
5. **critical-issues.md** - Update with:
   - New bugs discovered
   - Issues resolved
   - Technical debt items
6. **debug_log.md** - Append new entries with timestamps, never delete

## Related Resources

- Main package: `/home/user1/ROS2_Workspace/ros2_ws/src/cone_stellation/`
- GLIM reference: `/home/user1/ROS2_Workspace/GLIM_ws/src/glim/`
- Build artifacts: `../build/` and `../install/`