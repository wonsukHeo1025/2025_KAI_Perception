import os
import shutil
import re
from typing import Dict, Any, List

# ANSI Color codes
class Colors:
    GREEN = '\033[92m'      # GO 상태
    RED = '\033[91m'        # NO-GO 상태
    BLUE = '\033[94m'       # 헤더
    YELLOW = '\033[93m'     # 경고/중요 정보
    CYAN = '\033[96m'       # 섹션 헤더
    MAGENTA = '\033[95m'    # 값
    WHITE = '\033[97m'      # 기본 텍스트
    RESET = '\033[0m'       # 리셋
    BOLD = '\033[1m'        # 볼드
    DIM = '\033[2m'         # 흐림


def _term_size():
    try:
        size = shutil.get_terminal_size(fallback=(120, 40))
        return size.columns, size.lines
    except Exception:
        return 120, 40


def _status_bar(condition: bool) -> str:
    if condition:
        return f"{Colors.GREEN}{Colors.BOLD}● GO{Colors.RESET}"
    else:
        return f"{Colors.RED}{Colors.BOLD}● NO-GO{Colors.RESET}"


def _fmt_hz(hz: float) -> str:
    if hz > 0:
        return f"{Colors.WHITE}{hz:.1f}Hz{Colors.RESET}"
    else:
        return f"{Colors.DIM}0.0Hz{Colors.RESET}"


def _fmt_value(value: str) -> str:
    return f"{Colors.MAGENTA}{value}{Colors.RESET}"


# Visible-width helpers (ignore ANSI escape sequences when measuring/padding)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub('', s)

def _visible_len(s: str) -> int:
    return len(_strip_ansi(s))

def _truncate_visible(s: str, width: int) -> str:
    if width <= 0:
        return ''
    out: List[str] = []
    visible = 0
    i = 0
    while i < len(s) and visible < width:
        if s[i] == '\x1b':
            m = _ANSI_RE.match(s, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
        out.append(s[i])
        visible += 1
        i += 1
    return ''.join(out)

def _pad_visible(s: str, width: int) -> str:
    s2 = _truncate_visible(s, width)
    pad = width - _visible_len(s2)
    if pad > 0:
        s2 += ' ' * pad
    return s2


def render_dashboard(perception: Dict[str, Any], planning: Dict[str, Any], control: Dict[str, Any], safety: Dict[str, Any], nodes: Dict[str, Any]) -> str:
    cols, _ = _term_size()
    lines: List[str] = []

    # 메인 헤더
    header_sep = '═' * (cols - 2)
    lines.append(f"{Colors.CYAN}╔{header_sep}╗{Colors.RESET}")
    title = "AUTONOMOUS VEHICLE HEALTH MONITOR"
    title_padding = (cols - len(title) - 4) // 2
    lines.append(f"{Colors.CYAN}║{Colors.RESET}{' ' * title_padding}{Colors.BOLD}{Colors.WHITE}{title}{Colors.RESET}{' ' * (cols - len(title) - 4 - title_padding)}{Colors.CYAN}║{Colors.RESET}")
    lines.append(f"{Colors.CYAN}╠{header_sep}╣{Colors.RESET}")

    # 섹션 헬퍼 함수들
    def section_header(title: str, color: str = Colors.CYAN):
        lines.append(f"{color}║{Colors.RESET} {Colors.BOLD}{color}{title}{Colors.RESET}{' ' * (cols - len(title) - 4)}{color}║{Colors.RESET}")

    def section_separator(style: str = 'double'):
        if style == 'double':
            sep = '═' * (cols - 2)
            lines.append(f"{Colors.CYAN}╠{sep}╣{Colors.RESET}")
        elif style == 'single':
            sep = '─' * (cols - 2)
            lines.append(f"{Colors.BLUE}╟{sep}╢{Colors.RESET}")

    def format_item(name: str, key: str, data_dict: Dict, width: int = 20) -> str:
        data = data_dict.get(key)
        if not data:
            return f"{Colors.DIM}{name:<14}: ──────────{Colors.RESET}"

        status = _status_bar(data['ok'])
        hz = _fmt_hz(data['hz'])
        value_str = ""
        if data.get('value'):
            value_str = f" {_fmt_value(data['value'])}"

        return f"{name:<14}: {status} {hz}{value_str}"

    def row(items: List[str]):
        # 터미널 너비에 따라 열 수 결정
        if cols < 100:
            # 좁은 터미널: 2열
            if len(items) <= 2:
                if len(items) == 1:
                    text = items[0]
                else:
                    text = f"{items[0]} │ {items[1]}"
            else:
                # 3개 이상이면 2행으로 나누기
                row1 = f"{items[0]} │ {items[1]}"
                row2 = items[2] if len(items) == 3 else f"{items[2]} │ {items[3] if len(items) > 3 else ''}"
                # 고정된 너비로 패딩 (ANSI 무시)
                fixed_width = cols - 4
                row1_padded = _pad_visible(row1, fixed_width)
                row2_padded = _pad_visible(row2, fixed_width)
                lines.append(f"{Colors.CYAN}║{Colors.RESET} {row1_padded} {Colors.CYAN}║{Colors.RESET}")
                lines.append(f"{Colors.CYAN}║{Colors.RESET} {row2_padded} {Colors.CYAN}║{Colors.RESET}")
                return
        else:
            # 넓은 터미널: 3열
            if len(items) == 1:
                text = items[0]
            elif len(items) == 2:
                text = f"{items[0]} │ {items[1]}"
            else:
                col_width = (cols - 8) // 3
                text = f"{items[0]:<{col_width}} │ {items[1]:<{col_width}} │ {items[2]:<{col_width}}"

        # 고정된 너비로 패딩하여 모든 행의 오른쪽 끝을 일정하게 맞춤 (ANSI 무시)
        fixed_width = cols - 4
        text_padded = _pad_visible(text, fixed_width)

        lines.append(f"{Colors.CYAN}║{Colors.RESET} {text_padded} {Colors.CYAN}║{Colors.RESET}")

    # PERCEPTION 섹션
    section_header("PERCEPTION MODULE", Colors.GREEN)
    row([
        format_item('Camera 1', 'cam1', perception),
        format_item('Camera 2', 'cam2', perception),
        format_item('LiDAR', 'lidar', perception)
    ])
    row([
        format_item('Cones Lidar', 'cones_lidar', perception),
        format_item('Cones Fused', 'cones_fused', perception),
        format_item('Cones UKF', 'cones_ukf', perception)
    ])
    row([format_item('Odometry', 'odom', perception)])

    # PLANNING 섹션
    section_separator('double')
    section_header("PLANNING MODULE", Colors.YELLOW)
    row([
        format_item('Local Path', 'local_path', planning),
        format_item('Speed Profile', 'speed_profile', planning)
    ])

    # CONTROL 섹션
    section_separator('double')
    section_header("CONTROL MODULE", Colors.MAGENTA)
    row([
        format_item('Steer Cmd', 'steer_cmd', control),
        format_item('Steer FB', 'steer_fb', control),
        format_item('Speed FB', 'speed_fb', control)
    ])
    row([
        format_item('RPM Cmd', 'rpm_cmd', control),
        format_item('RPM Target', 'rpm_target', control),
        format_item('Throttle', 'throttle', control)
    ])

    # SAFETY 섹션
    section_separator('double')
    section_header("SAFETY SYSTEMS", Colors.RED)
    safety_items = []
    d = safety.get('aeb')
    if d:
        safety_items.append(format_item('AEB System', 'aeb', safety))
    else:
        safety_items.append(f"{Colors.DIM}AEB System : ───────{Colors.RESET}")

    if safety_items:
        row(safety_items)

    # NODES 섹션
    if nodes:
        section_separator('double')
        section_header("SYSTEM NODES", Colors.BLUE)

        node_items = []
        for name, st in nodes.items():
            status_color = Colors.GREEN if st.present else Colors.RED
            status_text = f"{status_color}●{'UP' if st.present else 'DOWN'}{Colors.RESET}"

            # 노드 이름을 15자로 제한
            display_name = name[:15] + "..." if len(name) > 15 else name.ljust(15)
            node_items.append(f"{Colors.WHITE}{display_name}{Colors.RESET}: {status_text}")

            # 2개씩 묶어서 행으로 표시
            if len(node_items) >= 2:
                text = f"{node_items[0]} │ {node_items[1]}"
                # 고정된 너비로 패딩 (ANSI 무시)
                fixed_width = cols - 4
                text_padded = _pad_visible(text, fixed_width)
                lines.append(f"{Colors.CYAN}║{Colors.RESET} {text_padded} {Colors.CYAN}║{Colors.RESET}")
                node_items = []

        # 남은 노드 처리
        if node_items:
            text = node_items[0]
            # 고정된 너비로 패딩 (ANSI 무시)
            fixed_width = cols - 4
            text_padded = _pad_visible(text, fixed_width)
            lines.append(f"{Colors.CYAN}║{Colors.RESET} {text_padded} {Colors.CYAN}║{Colors.RESET}")

    # 푸터
    lines.append(f"{Colors.CYAN}╚{header_sep}╝{Colors.RESET}")

    return "\n".join(lines)


def clear_screen():
    try:
        os.system('clear')
    except Exception:
        pass

