#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Veltric Translation Memory Experiment
--------------------------------------
5 blocks, each with:
  - 60s  baseline fixation (EEG baseline)
  - 180s encoding phase (word grid + audio)
  - 120s post-encoding rest (silence, fixation)
  - 24-question recall phase
  - 300s inter-block rest (fixation, silence)
    [skipped after final block]

Counterbalancing: 5 groups, each with a different
block order and list-condition pairing, read from
blocks_groupX.csv. Participant responses saved to
data/PXXX_responses.csv.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
from psychopy import visual, core, event, sound, gui, data, logging
import csv
import os
import sys
import serial

# ── Serial Port Trigger Setup ─────────────────────────────────────────────────
def open_trigger_port():
    try:
        port = serial.Serial('COM4', baudrate=115200, timeout=1)
        print("Trigger port opened on COM4")
        return port
    except Exception as e:
        print(f"WARNING: Could not open trigger port: {e}")
        print("Continuing without EEG triggers -- check COM port number")
        return None

def send_trigger(port, code):
    if port is None:
        return
    try:
        port.write(bytes([code]))
        core.wait(0.005)
        port.write(bytes([0]))
    except Exception as e:
        print(f"WARNING: Trigger {code} failed: {e}")

# Open the port once at experiment start
trigger_port = open_trigger_port()

# Wrapper so the rest of the script stays unchanged
def send_marker(code):
    send_trigger(trigger_port, code)
    print(f"MARKER {code} sent")
# ── Marker Code Reference ─────────────────────────────────────────────────────
# 1  = EXPERIMENT_START
# 2  = EXPERIMENT_END
# 10 = BLOCK_START
# 11 = BLOCK_END
# 21 = CONDITION_SILENCE
# 22 = CONDITION_WHITE_NOISE
# 23 = CONDITION_LOFI
# 24 = CONDITION_UNFAMILIAR_LYRICAL
# 25 = CONDITION_FAMILIAR_LYRICAL
# 30 = BASELINE_REST_START
# 31 = BASELINE_REST_END
# 40 = ENCODING_START
# 41 = AUDIO_START
# 42 = AUDIO_STOP
# 43 = ENCODING_END
# 50 = POST_ENCODE_REST_START
# 51 = POST_ENCODE_REST_END
# 60 = RECALL_START
# 61 = QUESTION_ONSET
# 62 = RESPONSE_CORRECT
# 63 = RESPONSE_INCORRECT
# 64 = RECALL_END
# 70 = RATING_SCREEN_START
# 71 = RATING_RESPONSE
# 72 = RATING_SCREEN_END
# 80 = INTER_BLOCK_REST_START
# 81 = INTER_BLOCK_REST_END

CONDITION_MARKERS = {
    'Silence':            21,
    'White noise':        22,
    'Lofi instrumental':  23,
    'Unfamiliar lyrical': 24,
    'Familiar lyrical':   25,
}

# ── Timing Configuration ──────────────────────────────────────────────────────
# All values in seconds. Change here to adjust timing globally.

BASELINE_REST_DURATION    = 30    # fixation cross before encoding (EEG baseline)
ENCODING_DURATION         = 90    # word grid study phase (3 minutes)
POST_ENCODE_REST_DURATION = 30    # fixation cross after encoding, before recall
INTER_BLOCK_REST_DURATION = 60    # fixation cross between blocks (5 minutes)

# For real data collection use:
# BASELINE_REST_DURATION    = 60
# ENCODING_DURATION         = 180
# POST_ENCODE_REST_DURATION = 120
# INTER_BLOCK_REST_DURATION = 300

# ── Experiment Info Dialog ────────────────────────────────────────────────────
exp_info = {
    'participant': '',
    'group': '1',
}

dlg = gui.DlgFromDict(
    dictionary=exp_info,
    title='Veltric Experiment',
    order=['participant', 'group'],
)
if not dlg.OK:
    core.quit()

participant = exp_info['participant'].strip()
group       = exp_info['group'].strip()

if not participant:
    print("ERROR: Participant number cannot be blank.")
    core.quit()

if group not in ['1', '2', '3', '4', '5']:
    print("ERROR: Group must be 1, 2, 3, 4, or 5.")
    core.quit()

# ── Paths ─────────────────────────────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
blocks_file  = os.path.join(script_dir, f"blocks_group{group}.csv")
data_dir     = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True)
output_file  = os.path.join(data_dir, f"P{participant}_group{group}_responses.csv")

# ── Load Block Order ──────────────────────────────────────────────────────────
blocks = []
with open(blocks_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        blocks.append(row)

# ── Window ────────────────────────────────────────────────────────────────────
win = visual.Window(
    size=[1440, 900],
    fullscr=True,
    screen=0,
    color='black',
    colorSpace='rgb',
    units='height',
    allowGUI=False,
)
win.mouseVisible = False

# ── Reusable Stimuli ──────────────────────────────────────────────────────────
fixation = visual.TextStim(
    win,
    text='+',
    height=0.08,
    color='white',
)

message_text = visual.TextStim(
    win,
    text='',
    height=0.04,
    color='white',
    wrapWidth=1.4,
    alignText='center',
)

timer_stim = visual.TextStim(
    win,
    text='',
    height=0.045,
    color='white',
    pos=(0, 0.43),
)

word_grid_stim = visual.TextStim(
    win,
    text='',
    height=0.026,
    color='white',
    font='Courier New',
    pos=(0, -0.02),
    anchorHoriz='center',
    anchorVert='center',
    wrapWidth=1.6,
)

question_stim = visual.TextStim(
    win,
    text='',
    height=0.05,
    color='white',
    pos=(0, 0.25),
    bold=True,
)

prompt_stim = visual.TextStim(
    win,
    text='Which Veltric word means this?',
    height=0.035,
    color='#aaaaaa',
    pos=(0, 0.14),
)

option_positions = [(0, 0.02), (0, -0.1), (0, -0.22), (0, -0.34)]
option_stims = [
    visual.TextStim(win, text='', height=0.038, color='white', pos=pos)
    for pos in option_positions
]

option_boxes = [
    visual.Rect(win, width=0.9, height=0.07, pos=pos,
                fillColor=None, lineColor='#444444', lineWidth=2)
    for pos in option_positions
]

selected_box = visual.Rect(
    win, width=0.9, height=0.07, pos=(0, 0),
    fillColor='#1a3a6b', lineColor='#4488ff', lineWidth=2,
)

progress_stim = visual.TextStim(
    win, text='', height=0.03, color='#666666', pos=(0, 0.42),
)

clock = core.Clock()

# ── Helper Functions ──────────────────────────────────────────────────────────

def show_message(text, wait_for_space=True, duration=None):
    """Display a centred message. Advance on SPACE or after duration seconds."""
    message_text.text = text
    clock.reset()
    while True:
        message_text.draw()
        win.flip()
        keys = event.getKeys(keyList=['space', 'escape'])
        if 'escape' in keys:
            save_and_quit()
        if wait_for_space and 'space' in keys:
            break
        if duration is not None and clock.getTime() >= duration:
            break


def show_fixation(duration):
    """Show fixation cross for exactly duration seconds."""
    clock.reset()
    while clock.getTime() < duration:
        fixation.draw()
        win.flip()
        keys = event.getKeys(keyList=['escape'])
        if keys:
            save_and_quit()


def build_word_grid(encoding_file):
    """Read encoding CSV and return formatted grid string."""
    pairs = []
    with open(encoding_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row['veltric'], row['english']))

    col1 = pairs[0:8]
    col2 = pairs[8:16]
    col3 = pairs[16:24]

    lines = []
    for i in range(8):
        v1, e1 = col1[i] if i < len(col1) else ("", "")
        v2, e2 = col2[i] if i < len(col2) else ("", "")
        v3, e3 = col3[i] if i < len(col3) else ("", "")
        c1 = f"{v1}  →  {e1}" if v1 else ""
        c2 = f"{v2}  →  {e2}" if v2 else ""
        c3 = f"{v3}  →  {e3}" if v3 else ""
        lines.append(f"{c1:<32}{c2:<32}{c3}")

    return "\n".join(lines)


def run_encoding(encoding_file, audio_file, condition, block_num):
    """
    Show the full word list with countdown timer.
    Audio plays during encoding and stops when timer ends.
    Sends ENCODING_START (40), AUDIO_START (41), AUDIO_STOP (42),
    ENCODING_END (43) markers.
    """
    grid_text = build_word_grid(encoding_file)
    word_grid_stim.text = grid_text

    # Load audio
    audio = None
    if condition.lower() != 'silence':
        audio_path = os.path.join(script_dir, audio_file)
        if os.path.exists(audio_path):
            audio = sound.Sound(audio_path, loops=-1)
        else:
            print(f"WARNING: Audio file not found: {audio_path}")

    # ── MARKER: encoding phase begins ────────────────────────────────────────
    send_marker(40)  # ENCODING_START

    # Start audio and send audio marker
    if audio:
        send_marker(41)  # AUDIO_START
        audio.play()

    clock.reset()
    duration = ENCODING_DURATION

    while clock.getTime() < duration:
        t = clock.getTime()
        remaining = max(0, duration - t)
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        timer_stim.text = f"{mins}:{secs:02d} remaining"
        timer_stim.color = 'red' if remaining <= 30 else 'white'

        timer_stim.draw()
        word_grid_stim.draw()
        win.flip()

        keys = event.getKeys(keyList=['escape'])
        if keys:
            if audio:
                audio.stop()
            save_and_quit()

    # Stop audio and send audio stop marker
    if audio:
        audio.stop()
        send_marker(42)  # AUDIO_STOP

    # ── MARKER: encoding phase ends ──────────────────────────────────────────
    send_marker(43)  # ENCODING_END

    show_message(
        "Study phase complete.\n\nPlease rest quietly.\n"
        "Do not rehearse the word pairs.",
        wait_for_space=False,
        duration=2,
    )


def run_recall(questions_file, block_num, condition):
    """
    Run 24 recall questions. Shows English word, participant
    selects Veltric translation using 1-4 keys.
    Sends RECALL_START (60), QUESTION_ONSET (61) per question,
    RESPONSE_CORRECT (62) or RESPONSE_INCORRECT (63) per response,
    and RECALL_END (64).
    Returns list of response dicts.
    """
    questions = []
    with open(questions_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)

    responses = []

    # ── MARKER: recall phase begins ──────────────────────────────────────────
    send_marker(60)  # RECALL_START

    for q_num, q in enumerate(questions, 1):
        english     = q['english']
        correct     = q['correct_veltric']
        options     = [q['option_1'], q['option_2'],
                       q['option_3'], q['option_4']]
        correct_pos = int(q['correct_pos'])

        for i, (opt, stim, box) in enumerate(
                zip(options, option_stims, option_boxes)):
            stim.text = f"{i+1}.  {opt}"
            box.lineColor = '#444444'
            box.fillColor = None

        question_stim.text = english
        progress_stim.text = f"Question {q_num} of {len(questions)}"

        selected     = None
        response_key = None
        rt           = None
        clock.reset()

        # ── MARKER: this question appears on screen ───────────────────────────
        send_marker(61)  # QUESTION_ONSET

        while True:
            progress_stim.draw()
            question_stim.draw()
            prompt_stim.draw()

            for i, (stim, box) in enumerate(zip(option_stims, option_boxes)):
                if selected is not None and selected == i:
                    selected_box.pos = option_positions[i]
                    selected_box.draw()
                else:
                    box.draw()
                stim.draw()

            win.flip()

            keys = event.getKeys(
                keyList=['1', '2', '3', '4', 'escape'],
                timeStamped=clock,
            )

            for key, timestamp in keys:
                if key == 'escape':
                    save_and_quit()
                if key in ['1', '2', '3', '4']:
                    selected     = int(key) - 1
                    response_key = key
                    rt           = timestamp

            if selected is not None:
                # Show highlight briefly
                progress_stim.draw()
                question_stim.draw()
                prompt_stim.draw()
                for i, (stim, box) in enumerate(
                        zip(option_stims, option_boxes)):
                    if selected == i:
                        selected_box.pos = option_positions[i]
                        selected_box.draw()
                    else:
                        box.draw()
                    stim.draw()
                win.flip()
                core.wait(0.3)
                break

        given_veltric = options[selected]
        is_correct    = 1 if given_veltric == correct else 0

        # ── MARKER: response — correct or incorrect ───────────────────────────
        if is_correct:
            send_marker(62)  # RESPONSE_CORRECT
        else:
            send_marker(63)  # RESPONSE_INCORRECT

        responses.append({
            'participant':     participant,
            'group':           group,
            'block_num':       block_num,
            'condition':       condition,
            'question_num':    q_num,
            'english':         english,
            'correct_veltric': correct,
            'given_veltric':   given_veltric,
            'correct_pos':     correct_pos,
            'response_key':    response_key,
            'is_correct':      is_correct,
            'rt':              round(rt, 4) if rt else '',
        })

    # ── MARKER: recall phase ends ─────────────────────────────────────────────
    send_marker(64)  # RECALL_END

    return responses


def save_responses(all_responses):
    """Write all collected responses to CSV."""
    if not all_responses:
        return
    fieldnames = list(all_responses[0].keys())
    file_exists = os.path.exists(output_file)
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(all_responses)


def save_responses_with_rating(block_responses, focus_rating, block_num):
    """
    Save block responses including focus rating to the main output file,
    and also append a summary row to a separate ratings file.
    """
    if not block_responses:
        return

    fieldnames = list(block_responses[0].keys())
    if 'focus_rating' not in fieldnames:
        fieldnames.append('focus_rating')

    ratings_output = output_file.replace('_responses.csv', '_with_ratings.csv')
    file_exists = os.path.exists(ratings_output)
    with open(ratings_output, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in block_responses:
            row_copy = dict(row)
            row_copy['focus_rating'] = focus_rating
            writer.writerow(row_copy)

    ratings_summary = os.path.join(
        data_dir, f"P{participant}_group{group}_ratings.csv"
    )
    file_exists = os.path.exists(ratings_summary)
    with open(ratings_summary, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['participant', 'group', 'block_num',
                        'condition', 'focus_rating']
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'participant':  participant,
            'group':        group,
            'block_num':    block_num,
            'condition':    condition,
            'focus_rating': focus_rating,
        })


def get_focus_rating(block_num, condition):
    """
    Display a 1-7 Likert scale focus question after each block.
    Sends RATING_SCREEN_START (70), RATING_RESPONSE (71),
    RATING_SCREEN_END (72) markers.
    Returns the rating as an integer.
    """
    question = visual.TextStim(
        win,
        text="During the last block, how focused were you\n"
             "on the vocabulary task?",
        height=0.045,
        color='white',
        pos=(0, 0.28),
        alignText='center',
    )

    scale_label_low = visual.TextStim(
        win,
        text="1\nNot at all\nfocused",
        height=0.032,
        color='#aaaaaa',
        pos=(-0.54, -0.05),
        alignText='center',
    )

    scale_label_high = visual.TextStim(
        win,
        text="7\nExtremely\nfocused",
        height=0.032,
        color='#aaaaaa',
        pos=(0.54, -0.05),
        alignText='center',
    )

    instruction = visual.TextStim(
        win,
        text="Press 1 – 7 to select your rating.",
        height=0.034,
        color='#888888',
        pos=(0, -0.32),
    )

    n_points = 7
    spacing  = 0.18
    start_x  = -(spacing * (n_points - 1) / 2)
    box_y    = 0.08

    boxes = []
    labels = []
    for i in range(n_points):
        x = start_x + i * spacing
        boxes.append(visual.Rect(
            win, width=0.12, height=0.12, pos=(x, box_y),
            fillColor=None, lineColor='#555555', lineWidth=2,
        ))
        labels.append(visual.TextStim(
            win, text=str(i + 1), height=0.042,
            color='white', pos=(x, box_y),
        ))

    # ── MARKER: rating screen appears ────────────────────────────────────────
    send_marker(70)  # RATING_SCREEN_START

    selected = None

    while True:
        question.draw()
        scale_label_low.draw()
        scale_label_high.draw()
        instruction.draw()

        for i, (box, label) in enumerate(zip(boxes, labels)):
            if selected is not None and selected == i:
                box.fillColor = '#1a3a6b'
                box.lineColor = '#4488ff'
            else:
                box.fillColor = None
                box.lineColor = '#555555'
            box.draw()
            label.draw()

        win.flip()

        keys = event.getKeys(
            keyList=['1', '2', '3', '4', '5', '6', '7', 'escape']
        )

        for key in keys:
            if key == 'escape':
                save_and_quit()
            if key in ['1', '2', '3', '4', '5', '6', '7']:
                selected = int(key) - 1

        if selected is not None:
            for i, (box, label) in enumerate(zip(boxes, labels)):
                if selected == i:
                    box.fillColor = '#1a3a6b'
                    box.lineColor = '#4488ff'
                else:
                    box.fillColor = None
                    box.lineColor = '#555555'
                box.draw()
                label.draw()
            question.draw()
            scale_label_low.draw()
            scale_label_high.draw()

            confirm = visual.TextStim(
                win,
                text=f"You selected: {selected + 1}\n\n"
                     f"Press SPACE to confirm or 1–7 to change.",
                height=0.035,
                color='#aaaaaa',
                pos=(0, -0.32),
            )
            confirm.draw()
            win.flip()

            while True:
                confirm_keys = event.getKeys(
                    keyList=['space', '1', '2', '3',
                             '4', '5', '6', '7', 'escape']
                )
                for k in confirm_keys:
                    if k == 'escape':
                        save_and_quit()
                    if k == 'space':
                        # ── MARKER: participant confirmed rating ──────────────
                        send_marker(71)  # RATING_RESPONSE
                        send_marker(72)  # RATING_SCREEN_END
                        return selected + 1
                    if k in ['1', '2', '3', '4', '5', '6', '7']:
                        selected = int(k) - 1
                        break
                else:
                    continue
                break


def save_and_quit():
    """Emergency save and exit."""
    print("Experiment interrupted — saving data.")
    if trigger_port is not None:
        trigger_port.close()
    win.close()
    core.quit()


# ── Experiment Start ──────────────────────────────────────────────────────────

# ── MARKER: experiment begins ─────────────────────────────────────────────────
send_marker(1)  # EXPERIMENT_START

# ── Instructions ──────────────────────────────────────────────────────────────
show_message(
    "Welcome to the experiment.\n\n"
    "In each block you will study a list of 24 word pairs\n"
    "from a made-up language called Veltric.\n\n"
    "You will have 3 minutes to learn as many translations\n"
    "as possible.\n\n"
    "After studying, there will be a short rest period.\n"
    "Then you will be tested on the translations.\n\n"
    "You will complete 5 blocks in total with breaks in between.\n\n"
    "Press SPACE to continue.",
    wait_for_space=True,
)

show_message(
    "During the recall test:\n\n"
    "You will see an English word and must select the correct\n"
    "Veltric translation using keys  1  2  3  4  on your keyboard.\n\n"
    "There is no time limit per question.\n"
    "Take your time and do your best.\n\n"
    "Press SPACE to begin the first block.",
    wait_for_space=True,
)

# ── Main Experiment Loop ──────────────────────────────────────────────────────

all_responses = []
n_blocks      = len(blocks)

for block_idx, block in enumerate(blocks):
    block_num  = int(block['block_num'])
    condition  = block['condition']
    list_file  = block['list_file']
    audio_file = block['audio_file']

    questions_file = os.path.join(script_dir, list_file)
    encoding_file  = os.path.join(
        script_dir,
        list_file.replace('_questions', '_encoding')
    )

    # ── MARKER: block begins ──────────────────────────────────────────────────
    send_marker(10)  # BLOCK_START

    # ── MARKER: which condition this block is ─────────────────────────────────
    send_marker(CONDITION_MARKERS[condition])

    # ── Block start message ───────────────────────────────────────────────────
    show_message(
        f"Block {block_num} of {n_blocks}\n\n"
        f"Condition: {condition}\n\n"
        "First there will be a short rest.\n"
        "Please sit quietly and look at the cross.\n\n"
        "Press SPACE when you are ready.",
        wait_for_space=True,
    )

    # ── 1. Baseline rest ──────────────────────────────────────────────────────
    send_marker(30)  # BASELINE_REST_START
    show_fixation(BASELINE_REST_DURATION)
    send_marker(31)  # BASELINE_REST_END

    # ── 2. Pre-encoding message ───────────────────────────────────────────────
    show_message(
        "The word list is about to appear.\n\n"
        "Study as many pairs as you can.\n"
        "You have 2 minutes.\n\n"
        "Press SPACE to start.",
        wait_for_space=True,
    )

    # ── 3. Encoding phase — markers sent inside run_encoding ─────────────────
    run_encoding(encoding_file, audio_file, condition, block_num)

    # ── 4. Post-encoding rest ─────────────────────────────────────────────────
    send_marker(50)  # POST_ENCODE_REST_START
    show_fixation(POST_ENCODE_REST_DURATION)
    send_marker(51)  # POST_ENCODE_REST_END

    # ── 5. Pre-recall message ─────────────────────────────────────────────────
    show_message(
        "Rest complete.\n\n"
        "The recall test is about to begin.\n"
        "Use keys  1  2  3  4  to select your answer.\n\n"
        "Press SPACE to start.",
        wait_for_space=True,
    )

    # ── 6. Recall phase — markers sent inside run_recall ─────────────────────
    block_responses = run_recall(questions_file, block_num, condition)
    all_responses.extend(block_responses)
    save_responses(block_responses)

    # ── 7. Subjective focus rating — markers sent inside get_focus_rating ─────
    focus_rating = get_focus_rating(block_num, condition)

    for r in block_responses:
        r['focus_rating'] = focus_rating

    save_responses_with_rating(block_responses, focus_rating, block_num)

    n_correct = sum(r['is_correct'] for r in block_responses)
    n_total   = len(block_responses)

    # ── MARKER: block ends ────────────────────────────────────────────────────
    send_marker(11)  # BLOCK_END

    # ── 8. Inter-block rest or end ────────────────────────────────────────────
    if block_idx < n_blocks - 1:
        show_message(
            f"Block {block_num} complete.\n\n"
            "Please rest for 1 minute.\n"
            "Sit quietly and relax.\n"
            "Do not rehearse the word pairs.\n\n"
            "The next block will begin automatically.",
            wait_for_space=False,
            duration=5,
        )
        send_marker(80)  # INTER_BLOCK_REST_START
        show_fixation(INTER_BLOCK_REST_DURATION)
        send_marker(81)  # INTER_BLOCK_REST_END
    else:
        show_message(
            f"Block {block_num} complete.\n\n"
            "You have completed all 5 blocks.",
            wait_for_space=False,
            duration=3,
        )

# ── End Screen ────────────────────────────────────────────────────────────────

show_message(
    "The experiment is now complete.\n\n"
    "Thank you for your participation.\n\n"
    "Please let the experimenter know you have finished.\n\n"
    "Press SPACE to exit.",
    wait_for_space=True,
)

# ── MARKER: experiment ends ───────────────────────────────────────────────────
send_marker(2)  # EXPERIMENT_END

if trigger_port is not None:
    trigger_port.close()
    print("Trigger port closed")
win.close()
core.quit()