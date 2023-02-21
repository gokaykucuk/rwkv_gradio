"""
RWKV RNN Model - Gradio Space for HuggingFace
YT - Mean Gene Hacks - https://www.youtube.com/@MeanGeneHacks
(C) Gene Ruebsamen - 2/7/2023

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import gradio as gr
import codecs
from ast import literal_eval
from datetime import datetime
from rwkvstic.load import RWKV
from config import config, title
import torch
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_md(text):
    return text.replace("\n", "<br />")


def get_model():
    global model
    model = RWKV(
        **config
    )
    return model


model = None


def infer(
        prompt,
        mode="generative",
        max_new_tokens=10,
        temperature=0.1,
        top_p=1.0,
        end_adj=0.0,
        stop="<|endoftext|>",
        seed=42,
):
    global model

    if model is None:
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        model = get_model()

    max_new_tokens = int(max_new_tokens)
    temperature = float(temperature)
    top_p = float(top_p)
    stop = [x.strip(' ') for x in stop.split(',')]

    assert 1 <= max_new_tokens <= 384
    assert 0.0 <= temperature <= 1.0
    assert 0.0 <= top_p <= 1.0
    assert -999 <= end_adj <= 0.0

    temperature = max(0.05, temperature)
    if prompt == "":
        prompt = " "

    # Clear model state for generative mode
    model.resetState()
    if mode == "Q/A":
        prompt = f"Ask Expert\n\nQuestion:\n{prompt}\n\nExpert Full Answer:\n"

    print(f"PROMPT ({datetime.now()}):\n-------\n{prompt}")
    print(f"OUTPUT ({datetime.now()}):\n-------\n")
    # Load prompt
    model.loadContext(newctx=prompt)
    generated_text = ""
    done = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            char = model.forward(stopStrings=stop, temp=temperature, top_p_usual=top_p, end_adj=end_adj)[
                "output"]
            print(char, end='', flush=True)
            generated_text += char
            generated_text = generated_text.lstrip("\n ")

            for stop_word in stop:
                stop_word = codecs.getdecoder("unicode_escape")(stop_word)[0]
                if stop_word != '' and stop_word in generated_text:
                    done = True
                    break
            yield generated_text
            if done:
                print("<stopped>\n")
                break

    # print(f"{generated_text}")

    for stop_word in stop:
        stop_word = codecs.getdecoder("unicode_escape")(stop_word)[0]
        if stop_word != '' and stop_word in generated_text:
            generated_text = generated_text[:generated_text.find(stop_word)]

    gc.collect()
    yield generated_text


def chat(
        prompt,
        history,
        username,
        max_new_tokens=10,
        temperature=0.1,
        top_p=1.0,
        end_adj=0.0,
        seed=42,
):
    global model
    history = history or []

    intro = ""

    if model is None:
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        model = get_model()

    username = username.strip()
    username = username or "USER"

    intro = f'''The following is a verbose and detailed conversation between an AI assistant called FRITZ, and a human user called USER. FRITZ is intelligent, knowledgeable, wise and polite.

    {username}: What year was the french revolution?
    FRITZ: The French Revolution started in 1789, and lasted 10 years until 1799.
    {username}: 3+5=?
    FRITZ: The answer is 8.
    {username}: What year did the Berlin Wall fall?
    FRITZ: The Berlin wall stood for 28 years and fell in 1989.
    {username}: solve for a: 9-a=2
    FRITZ: The answer is a=7, because 9-7 = 2.
    {username}: wat is lhc
    FRITZ: The Large Hadron Collider (LHC) is a high-energy particle collider, built by CERN, and completed in 2008. It was used to confirm the existence of the Higgs boson in 2012.
    {username}: Tell me about yourself.
    FRITZ: My name is Fritz. I am an RNN based Large Language Model (LLM).
    '''

    if len(history) == 0:
        # no history, so lets reset chat state
        model.resetState()
        history = [[], model.emptyState]
        print("reset chat state")
    else:
        if history[0][0][0].split(':')[0] != username:
            model.resetState()
            history = [[], model.emptyState]
            print("username changed, reset state")
        else:
            model.setState(history[1])
            intro = ""

    max_new_tokens = int(max_new_tokens)
    temperature = float(temperature)
    top_p = float(top_p)

    assert 1 <= max_new_tokens <= 384
    assert 0.0 <= temperature <= 1.0
    assert 0.0 <= top_p <= 1.0
    assert -999 <= end_adj <= 0.0

    temperature = max(0.05, temperature)

    prompt = f"{username}: " + prompt + "\n"
    print(f"CHAT ({datetime.now()}):\n-------\n{prompt}")
    print(f"OUTPUT ({datetime.now()}):\n-------\n")
    # Load prompt

    model.loadContext(newctx=intro + prompt)

    out = model.forward(number=max_new_tokens, stopStrings=[
        "<|endoftext|>", username + ":"], temp=temperature, top_p_usual=top_p, end_adj=end_adj)

    generated_text = out["output"].lstrip("\n ")
    generated_text = generated_text.rstrip(username + ":")
    print(f"{generated_text}")

    gc.collect()
    history[0].append((prompt, generated_text))
    return history[0], [history[0], out["state"]]


examples = [
    [
        # Question Answering
        '''What is the capital of Germany?''', "Q/A", 25, 0.2, 0.8, 0.0, "<|endoftext|>,\\n"],
    [
        # Question Answering
        '''Are humans good or bad?''', "Q/A", 150, 0.8, 0.8, -1.0, "<|endoftext|>"],
    [
        # Question Answering
        '''What is the purpose of Vitamin A?''', "Q/A", 60, 0.2, 0.8, -0.5, "<|endoftext|>"],
    [
        # Chatbot
        '''This is a conversation between two AI large language models named Alex and Fritz. They are exploring each other's capabilities, and trying to ask interesting questions of one another to explore the limits of each others AI.

Conversation:
Alex: Good morning, Fritz, what type of LLM are you based upon?
Fritz: Morning Alex, I am an RNN with transformer level performance. My language model is 100% attention free.
Alex:''', "generative", 220, 0.9, 0.9, -3.0, "<|endoftext|>,\\n\\n"],
    [
        # Generate List
        '''Task given:

Please Write a Short story about a cat learning python

Best Full Response:
''', "generative", 140, 0.85, 0.8, -3.5, "<|endoftext|>"],
    [
        # Natural Language Interface
        '''Here is a short story (in the style of Tolkien) in which Aiden attacks a robot with a sword:
        ''', "generative", 140, 0.85, 0.8, -3.5, "<|endoftext|>"]
]

iface = gr.Interface(
    fn=infer,
    description=f'''<h3>Generative and Question/Answer</h3>''',
    allow_flagging="never",
    inputs=[
        gr.Textbox(lines=20, label="Prompt"),  # prompt
        gr.Radio(["generative", "Q/A"],
                 value="generative", label="Choose Mode"),
        gr.Slider(1, 256, value=40),  # max_tokens
        gr.Slider(0.0, 1.0, value=0.8),  # temperature
        gr.Slider(0.0, 1.0, value=0.85),  # top_p
        gr.Slider(-99, 0.0, value=0.0, step=0.5, label="Reduce End of Text Probability"),  # end_adj
        gr.Textbox(lines=1, value="<|endoftext|>")  # stop
    ],
    outputs=gr.Textbox(label="Generated Output", lines=25),
    examples=examples,
    cache_examples=False,
).queue()

chatiface = gr.Interface(
    fn=chat,
    description=f'''<h3>Chatbot</h3><h4>Refresh page or change name to reset memory context</h4>''',
    allow_flagging="never",
    inputs=[
        gr.Textbox(lines=5, label="Message"),  # prompt
        "state",
        gr.Text(lines=1, value="USER", label="Your Name",
                placeholder="Enter your Name"),
        gr.Slider(1, 256, value=60),  # max_tokens
        gr.Slider(0.0, 1.0, value=0.8),  # temperature
        gr.Slider(0.0, 1.0, value=0.85),  # top_p
        gr.Slider(-99, 0.0, value=-2, step=0.5, label="Reduce End of Text Probability"),  # end_adj
    ],
    outputs=[gr.Chatbot(label="Chat Log", color_map=(
        "green", "pink")), "state"],
).queue()

demo = gr.TabbedInterface(

    [iface, chatiface], ["Generative", "Chatbot"],
    title=title,

)

demo.queue()
demo.launch(share=True)
