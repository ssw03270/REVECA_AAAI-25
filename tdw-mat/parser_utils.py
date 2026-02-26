import re
import random
import difflib

def parse_name_id(s: str):
    match = re.search(r"<([^<>]+)>\s*\((\d+)\)", s.strip())
    if match:
        name, value = match.groups()
        return name, int(value)
    return None, None


def relevance_to_language(relevance_score):
    result = ""
    if relevance_score == 1:
        result = "Strong relevance"
    elif relevance_score == 0.5:
        result = "Medium relevance"
    elif relevance_score == 0:
        result = "Low relevance"
    elif relevance_score == -1:
        result = "None relevance"

    return result

def parse_validate_request(answer, rooms):
    _name = answer.split('Collaborator name:')[-1].split(', Select:')[0].strip()
    _select = answer.split('Select:')[-1].split(', Action completed:')[0].strip()
    _answer = answer.split('Action completed:')[-1].strip()

    _select, _ = parse_rooms_selection(answer, rooms)
    return _name, _select, _answer


def build_room_choices(rooms):
    """
    Build option strings like 'A. livingroom' from the room list.
    """
    if len(rooms) > 26:
        raise ValueError("Alphabet labeling (A-Z) cannot be used when rooms has more than 26 items.")
    return "\n".join(f"{chr(65+i)}. {room}" for i, room in enumerate(rooms))


def _norm(s: str) -> str:
    # Normalize by removing whitespace and lowercasing.
    return re.sub(r"\s+", "", s).lower()

def _find_room_by_exact_name(token: str, rooms):
    norm = _norm(token)
    for i, r in enumerate(rooms):
        if _norm(r) == norm:
            return rooms[i], i
    return None, None

def _find_room_by_fuzzy(token: str, rooms, cutoff=0.55):
    # Pick the closest room name via difflib.
    candidates = difflib.get_close_matches(token, rooms, n=1, cutoff=cutoff)
    if candidates:
        best = candidates[0]
        idx = next(i for i, r in enumerate(rooms) if r == best)
        return best, idx
    return None, None

def _find_room_by_appearance(text: str, rooms):
    # If room names appear in text, choose the earliest one (ignoring spaces).
    t = _norm(text)
    best_idx = None
    best_pos = None
    for i, r in enumerate(rooms):
        p = t.find(_norm(r))
        if p != -1 and (best_pos is None or p < best_pos):
            best_pos = p
            best_idx = i
    if best_idx is not None:
        return rooms[best_idx], best_idx
    return None, None

def parse_rooms_selection(text: str, rooms):
    """
    Always returns (room, idx).
    Priority:
    1) Alphabet/number/room name in the 'Select:' field
    2) If no 'Select:', a single alphabet token (A-Z) in the body
    3) Exact room name found in the body
    4) Fuzzy matching on the 'Select:' token
    5) Fuzzy matching on the entire body
    6) Random fallback
    """
    if not rooms:
        return None, None  # Cannot select if rooms is empty.

    # 1) Extract token from the Select: field.
    m = re.search(r"select\s*:\s*([A-Za-z]|\d+|[^,\n]+)", text, re.IGNORECASE)
    token = m.group(1).strip() if m else None

    # a) Single alphabet character.
    if token and len(token) == 1 and token.isalpha():
        idx = ord(token.upper()) - 65
        if 0 <= idx < len(rooms):
            return rooms[idx], idx

    # b) Number (1-based).
    if token and token.isdigit():
        idx = int(token) - 1
        if 0 <= idx < len(rooms):
            return rooms[idx], idx

    # c) Exact room name match.
    if token:
        r, i = _find_room_by_exact_name(token, rooms)
        if r is not None:
            return r, i

    # 2) Single alphabet token in the body.
    m2 = re.search(r"\b([A-Za-z])\b", text)
    if m2:
        idx = ord(m2.group(1).upper()) - 65
        if 0 <= idx < len(rooms):
            return rooms[idx], idx

    # 3) Exact room name appearing in the body.
    r, i = _find_room_by_appearance(text, rooms)
    if r is not None:
        return r, i

    # 4) Fuzzy match on the token.
    if token:
        r, i = _find_room_by_fuzzy(token, rooms)
        if r is not None:
            return r, i

    # 5) Fuzzy match on the full body.
    #    Compare similarity for each room and keep the highest score.
    best_room, best_idx, best_score = None, None, 0.0
    for i, rname in enumerate(rooms):
        score = difflib.SequenceMatcher(None, _norm(text), _norm(rname)).ratio()
        if score > best_score:
            best_score = score
            best_room, best_idx = rname, i
    if best_room is not None and best_score >= 0.4:  # If score is too low, fall back to random.
        return best_room, best_idx

    # 6) Random fallback.
    idx = random.randrange(len(rooms))
    return rooms[idx], idx


def extract_message(text: str) -> str:
    # Count quote characters.
    quote_count = text.count('"')

    # If there are exactly two quotes,
    if quote_count == 2:
        # capture only the first quoted segment.
        match = re.search(r'"([^"]*)"', text)
        if match:
            return match.group(1)

    # Otherwise return the full input string.
    return text
