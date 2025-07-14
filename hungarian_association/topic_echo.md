❯ ros2 topic echo /sorted_cones_time --once
header:
  stamp:
    sec: 1749793044
    nanosec: 649225591
  frame_id: os_sensor
class_names:
- Unknown
- Unknown
- Unknown
- Unknown
- Unknown
- Unknown
- Unknown
- Unknown
- Unknown
layout:
  dim:
  - label: ''
    size: 9
    stride: 27
  - label: ''
    size: 3
    stride: 3
  data_offset: 0
data:
- 0.5153380632400513
- -2.228254556655884
- -0.7074131369590759
- 0.6371853947639465
- 2.8125312328338623
- -0.7602007389068604
- 2.1677675247192383
- -2.3085362911224365
- -0.8307976126670837
- 2.2249550819396973
- 2.80653977394104
- -0.8366633057594299
- 3.496898651123047
- -2.3341033458709717
- -0.9482884407043457
- 3.629138231277466
- 2.7774910926818848
- -0.9263086318969727
- 5.166191101074219
- 3.7858219146728516
- -0.916720449924469
- 5.256827354431152
- -2.297452926635742
- -0.938811719417572
- 6.642374515533447
- -1.3919013738632202
- -1.087106466293335
---

❯ ros2 topic echo /fused_sorted_cones_ukf --once
header:
  stamp:
    sec: 1749793055
    nanosec: 998173674
  frame_id: os_sensor
cones:
- track_id: 899
  position:
    x: 0.1425809730821186
    y: -1.6495881014639302
    z: -0.4706347857085982
  color: Yellow cone
- track_id: 904
  position:
    x: 1.145999086146173
    y: -1.9107929363063918
    z: -0.6547366616123653
  color: Yellow cone
- track_id: 905
  position:
    x: 2.3809709854971226
    y: -2.705219201528228
    z: -0.8801699777997006
  color: Yellow cone
- track_id: 906
  position:
    x: 3.6688547368692332
    y: -3.3769505510657134
    z: -0.996886383456449
  color: Yellow cone
- track_id: 924
  position:
    x: 0.11135276117026893
    y: 3.5895793223241634
    z: -0.6165335442987048
  color: Blue cone
- track_id: 925
  position:
    x: 1.7216429274319798
    y: 3.3837946858505252
    z: -0.7936762383220569
  color: Blue cone
- track_id: 926
  position:
    x: 3.359241003041047
    y: 2.563215007888813
    z: -0.9578894739899271
  color: Blue cone
- track_id: 928
  position:
    x: 4.964823008456119
    y: 1.5642846691838252
    z: -0.9930210458686038
  color: Blue cone
- track_id: 929
  position:
    x: 4.832318955058161
    y: -4.075786624272047
    z: -0.9698321020874033
  color: Yellow cone
- track_id: 930
  position:
    x: 6.728972791184039
    y: 0.5493932117331444
    z: -1.076593883566871
  color: Blue cone
- track_id: 933
  position:
    x: 6.1233861972168055
    y: -4.867227677488286
    z: -1.0589736425090908
  color: Yellow cone
---
