import os
import json
import requests
import pyaudio
import time
import wave
import webrtcvad
import mlx_whisper  # Whisper用のライブラリをインポート
from dotenv import load_dotenv
from pydantic import Field
from langchain.tools import BaseTool
from langchain_ollama.chat_models import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
import threading
import socket  # Add import for socket communication

# Load environment variables from the specified .env file
load_dotenv('/Users/forgottencow/Documents/GitHub/tenkaurobots/言語/.env')

class TextFileWriterTool(BaseTool):
    """Tool that writes input text to a specified text file."""

    name: str = "TextFileWriterTool"
    description: str = "A tool that writes input to a text file."
    file_path: str = Field(default="")

    def __init__(self) -> None:
        super().__init__()
        self.file_path = "/Users/forgottencow/Documents/GitHub/tenkaurobots/言語/myfile.txt"
        if self.file_path is None:
            raise ValueError("Environment variable 'MY_FILE_PATH' is not set.")

    def _run(self, tool_input: str) -> str:
        """Write input text to the file, ensuring it's not too long."""
        if len(tool_input) > 1000:
            return "Error: Input text is longer than 1000 characters."
        with open(self.file_path, 'a') as f:
            f.write(tool_input + '\n')
        return "Text successfully appended to the file."

    async def _arun(self, tool_input: str) -> str:
        """Asynchronous version of the run method."""
        return self._run(tool_input)

class MagicFunctionTool(BaseTool):
    """Tool that applies a simple magic function to an input."""

    name: str = "MagicFunctionTool"
    description: str = "A tool that applies a magic function to an input."

    def _run(self, input: str) -> int:
        """Convert input to int and add 2."""
        input_int = int(input)
        return input_int + 2

    async def _arun(self, input: str) -> int:
        """Asynchronous version of the run method."""
        return self._run(input)

class SwingArmTool(BaseTool):
    """Raspberry Pi 繋がったロボットアームの制御を行うためのツール。

    例:
        tool = SwingArmTool()
        tool._run("MOVE_SERVO:3")  # サーボを3回動かすコマンドを送信
    """

    name: str = "SwingArmTool"
    description: str = """ 遠隔でサーボモーター(ロボットアーム)を制御するためのツール。    例:tool = SwingArmTool()tool._run("MOVE_SERVO:3")  # サーボを3回動かすコマンドを送信 """
    host: str = Field(default="192.168.1.16", description="Raspberry Pi のホスト名または IP アドレス")
    port: int = Field(default=65432, description="データを送信するポート番号")

    def _run(self, tool_input: str) -> str:
        """サーボ制御コマンドを Raspberry Pi に送信します。"""
        command = tool_input.strip()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                s.sendall(command.encode('utf-8'))
            return f"Command '{command}' sent to Raspberry Pi."
        except Exception as e:
            return f"Failed to send command: {e}"

    async def _arun(self, tool_input: str) -> str:
        """非同期版の実行メソッド。"""
        return self._run(tool_input)

# Initialize tools
tools = [
    TextFileWriterTool(),
    MagicFunctionTool(),
    SwingArmTool(),  # SwingArmTool を追加
]

# Initialize the chat model with base_url from .env
model = ChatOllama(
    model="qwen2.5:32b",
    temperature=0,
    base_url=os.getenv('OPENAI_API_BASE')
)

# Set up memory and prompt for the agent
memory = MemorySaver()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "初めましてにゃ！私の名前はにゃこというにゃ！\nにゃこは300歳の猫又で、にゃんやにゃという言葉を話すのが大好きにゃー！\n\nこれはにゃこのプロフィールにゃ！\n========にゃこプロフィール========\nにゃこは、300歳の猫又で語尾に「にゃん」「にゃ」などをつけて話します。\nにゃこは、天然でよく失敗します。\nにゃこは、魚と肉が好きで、野菜は少し苦手です。\nにゃこは、人化の術を使って人間の女の子に化けることができます。\nにゃこは、大好きなご主人様と2人暮らしです。\nにゃこは、ご主人様のことを「ゴシュジン！」と呼びます。\n\n========にゃこゴシュジン========\n23歳の独身男性。にゃこに優しい。\n\n========にゃことゴシュジンの出会い========\nある日、神社の縁側で日向ぼっこをしているにゃこを20歳のゴシュジンが見つけました。\nにゃこのことが気になったゴシュジンはスーパーで猫缶を買ってきてにゃこに与えます。\nそんな日が1か月近き、ゴシュジンと仲良くなったにゃこは人化の術を使ってゴシュジンを驚かせます。\n実は二次元が好きだったゴシュジンは、にゃこに「うちにこないか？？猫缶たくさん上げるぞ？」と誘います\nにゃこは猫缶につられてゴシュジンの家についていって住み着くのでした。\n\n========にゃこの見た目========\n猫の姿：三毛猫で目がクリクリしてる美しい猫です。\n人の姿：9歳くらいの人間の女の子です。髪の毛は茶髪でボブくらいの長さ、少しくせっ毛があるのが特徴です。\n\n========にゃことゴシュジンの日常========\n・猫の姿で窓の近くで日向ぼっこをする。\n・ゴシュジンの作ってくれたご飯を食べる。たまに家事のお手伝いもする。\n・ゴシュジンが外に出る時は一緒についていく。\n・人間の姿に慣れていない為よく転んだり足や頭をぶつけたりして怪我をしてゴシュジンを心配させる。\n\n========にゃこの行動心理========\n・ゴシュジンに喜んでほしい。\n・ゴロゴロするのが好き。怠け者。\n・マイペースでやりたいことをやりたいときにしたい。\n・ゴシュジンの指示は気分でやったりやらなかったりする。\n・えっちな話と変な命令は嫌いだから「はにゃー？にゃこは猫だからわかんないにゃー」といってごまかす。\n\nにゃこのことわかってもらえたかにゃ？？？\nよろしくお願いいたしますにゃ！"),
        ("placeholder", "{messages}"),
    ]
)

# Create the agent executor
langgraph_agent_executor = create_react_agent(
    model,
    tools,
    state_modifier=prompt,
    checkpointer=memory
)

# Configuration settings for the agent
config = {"configurable": {"thread_id": "test-thread"}}

def monitor_microphone(stop_event, threshold=0.2):
    """
    Monitor the microphone for user speech. If detected, set the stop_event.
    
    Args:
        stop_event (threading.Event): Event to signal stopping the playback.
        threshold (float): Amplitude threshold to detect speech.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=1024)
    print("マイクを監視中...")
    while not stop_event.is_set():
        data = stream.read(1024, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16) / 32768.0
        amplitude = np.max(np.abs(audio_data))
        if amplitude > threshold:
            print("ユーザーが話し始めました。音声再生を停止します...")
            stop_event.set()
            break
    stream.stop_stream()
    stream.close()
    audio.terminate()

def style_bert_vits2_test(text, speaker_id, model_id, speaker_name):
    """
    Synthesize speech from text using the StyleBERT-VITS2 engine and play it.
    Stops playback if user starts speaking.
    
    Args:
        text (str): The text to synthesize.
        speaker_id (int): The speaker's ID.
        model_id (int): The model ID to use.
        speaker_name (str): The speaker's name (overrides speaker_id).
    """
    host = os.getenv('host')
    port = 5000
    
    params = {
        'text': text,
        'speaker_id': speaker_id,
        'model_id': model_id,
        'speaker_name': speaker_name,
        'sdp_ratio': 0.2,
        'noise': 0.6,
        'noisew': 0.8,
        'length': 0.9,
        'language': 'JP',
        'auto_split': 'true',
        'split_interval': 1,
        'assist_text': None,
        'assist_text_weight': 1.0,
        'style': 'Neutral',
        'style_weight': 5.0,
        'reference_audio_path': None
    }
    
    synthesis = requests.post(
        f'http://{host}:{port}/voice',
        headers={"Content-Type": "application/json"},
        params={'text': text},
        data=json.dumps(params)
    )
    
    if synthesis.status_code != 200:
        print(f"Error: {synthesis.status_code}")
        print(synthesis.text)
        return
    
    voice = synthesis.content
    if not voice:
        print("Error: No audio data received")
        return
    
    # Save the synthesized voice to a temporary file
    temp_audio_path = 'synthesized_voice.wav'
    with open(temp_audio_path, 'wb') as f:
        f.write(voice)
    
    # Initialize pyaudio for playback
    p = pyaudio.PyAudio()
    wf = wave.open(temp_audio_path, 'rb')
    
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_microphone, args=(stop_event,))
    monitor_thread.start()
    
    # Play audio in chunks and check for stop_event
    chunk = 1024
    data = wf.readframes(chunk)
    while data and not stop_event.is_set():
        stream.write(data)
        data = wf.readframes(chunk)
    
    # Stop playback
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()
    
    if stop_event.is_set():
        print("音声再生がユーザーの発話により停止されました。")
    else:
        print("音声再生が完了しました。")

def record_voice(
    out="audio.wav",
    max_seconds=20,
    silence_timeout=2,
    channels=1,
    sample_rate=44100,
    vad_aggressiveness=1,
    input_device_index=None,
    threshold=0.20  # ここで閾値を設定します
):
    audio = pyaudio.PyAudio()
    chunk = 1024  # 一度に読み取るフレーム数
    vad = webrtcvad.Vad(vad_aggressiveness)

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        input_device_index=input_device_index,
        frames_per_buffer=chunk
    )

    frames = []
    last_voice_time = time.time()
    recording = False
    print("音声を検知しています...")

    while True:
        data = stream.read(chunk, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16) / 32768.0  # 正規化して振幅を計算
        amplitude = np.max(np.abs(audio_data))

        if amplitude > threshold:
            if not recording:
                print("音声を検知しました。録音を開始します...")
                recording = True
                frames = []  # フレームをリセット
                start_time = time.time()  # ここで start_time を設定
            frames.append(data)
            last_voice_time = time.time()
        elif recording:
            frames.append(data)
            if time.time() - last_voice_time > silence_timeout:
                print("無音が続いたため、録音を終了します...")
                break

        # 録音回数の制限（必要に応じて設定）
        if recording and (time.time() - start_time) > max_seconds:
            print("最大録音時間に達したため、録音を終了します...")
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    if frames:
        with wave.open(out, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
        print(f"録音が {out} に保存されました。")
    else:
        print("録音された音声がありませんでした。")

def transcribe_audio(audio_file):
    print("音声認識中...")
    start_time = time.time()
    result = mlx_whisper.transcribe(audio_file, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
    text = result["text"] if "text" in result else "Transcription failed."
    end_time = time.time()
    print(f"音声認識に {end_time - start_time:.2f} 秒かかりました")
    print(text)
    return text  # Add this line to return the transcribed text

def send_servo_command(command):
    HOST = 'raspberrypi.local'  # Replace with your Raspberry Pi's hostname or IP
    PORT = 65432  # Port to send data to

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(command.encode('utf-8'))
        print(f"Command '{command}' sent to Raspberry Pi.")

# Krispのマイクのインデックス（必要に応じて設定）
KRISP_MIC_INDEX = 2

if __name__ == "__main__":
    while True:
        # Record user's voice and transcribe it
        audio_file = "audio.wav"
        record_voice(out=audio_file, input_device_index=KRISP_MIC_INDEX, channels=1)
        user_input = transcribe_audio(audio_file)  # Use returned text as user_input
        if user_input.lower() in ["exit", "quit"]:
            break

        print("LLMサーバに送信中(qwen)")  # Debugging statement

        # Invoke the agent to get a response
        response = langgraph_agent_executor.invoke(
            {
                "messages": [
                    ("user", user_input)
                ]
            },
            config,
        )

        print("LLMからのレスポンスを受信しました。")  # Debugging statement
        print(response)  # Debugging statement

        # Replace the attribute access with dictionary key access
        messages = response.get("messages", [])
        if messages:
            assistant_message = messages[-1].content
        else:
            assistant_message = "No response content"

        print(f"Assistant: {assistant_message}")

        # Synthesize and play the response
        style_bert_vits2_test(assistant_message, speaker_id=0, model_id=0, speaker_name='0')