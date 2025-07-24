# rqt_console Settings Guide

## Why Loop Closure Logs Don't Appear in rqt_console

The loop closure logs are INFO level messages that may be filtered out by default rqt_console settings.

## How to View Loop Closure Logs in rqt_console

### 1. Check Log Level Filter
- Open rqt_console
- Look at the top toolbar for log level buttons
- Make sure **Info** is enabled (not just Warn/Error)
- The button should be highlighted/pressed

### 2. Logger Name Filter
If you want to see only loop closure related logs:
- In the filter section, add a logger name filter
- Enter one of these logger names:
  - `loop_closure` - for loop closure detector logs
  - `cone_mapping` - for SLAM mapping logs including loop closure factors

### 3. Message Content Filter
To search for specific loop closure events:
- Add a message content filter
- Use these keywords:
  - `"Loop closure detected"` - when loop closure is found
  - `"loop closure factor"` - when factor is added
  - `"FIND_CANDIDATES"` - for candidate search logs
  - `"VALIDATE"` - for validation process logs

### 4. Common Issues

#### All INFO logs are hidden by default
- Solution: Click the "Info" button in the toolbar to enable INFO level logs

#### Too many logs to find loop closure
- Solution: Use logger name filter with `loop_closure` or `cone_mapping`

#### Can't see any logs at all
- Check that the node is running: `ros2 node list`
- Check that rqt_console is connected to the right ROS domain

### 5. Alternative: Terminal Filtering
If rqt_console is problematic, you can filter terminal output:
```bash
ros2 run cone_stellation cone_slam_node 2>&1 | grep -E "loop_closure|Loop closure"
```

## Loop Closure Log Messages to Look For

1. **When loop closure is detected:**
   ```
   [cone_mapping]: Loop closure detected! X candidates found
   ```

2. **When loop closure factor is added:**
   ```
   [cone_mapping]: Added loop closure factor: X92 -> X120 with Y cone matches
   ```

3. **Debug messages (if debug logging enabled):**
   ```
   [loop_closure]: [DETECT_LOOP] START
   [loop_closure]: [FIND_CANDIDATES] Found X scored candidates
   [loop_closure]: [VALIDATE] Validating loop closure between frames X and Y
   ```