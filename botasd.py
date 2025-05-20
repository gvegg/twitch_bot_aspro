import twitchio
from twitchio.ext import commands
import asyncio
import json
from pathlib import Path
from collections import deque
import uuid
import logging
import urllib.parse
import random
import base64
import re
import aiohttp
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
import subprocess
import asyncio
from typing import Optional


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TwitchBot")

# Конфигурация API
API_CONFIG = {
    "text_base_url": "https://text.pollinations.ai",
    "image_base_url": "https://image.pollinations.ai",
    "text_models": ["openai-large", "searchgpt"],
    "default_text_model": "openai-large",
    "image_settings": {
        "width": 1920,
        "height": 1080,
        "model": "flux",
        "nologo": "true",
        "enhance": "true"
    },
    "channel": "uncle_biz",
    "max_image_size": 4 * 7680 * 4320,
    "allowed_formats": ["jpg", "jpeg", "png", "webp", "webm"],
    "vision_model": "openai-vision",
    "max_length": 450,
    "merge_threshold": 50,
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "silence_threshold": -40,
        "min_silence_len": 500,
        "keep_silence": 200,
        "model_size": "tiny",
        "chunk_duration": 5
    }
}

class AudioProcessor:
    def __init__(self, bot):
        self.bot = bot
        self.is_listening = False
        self.process = None
        self.lock = asyncio.Lock()
        self.session = aiohttp.ClientSession()  # Создаем сессию для запросов

    async def _transcribe_and_respond(self, audio_data: bytes):
        temp_file = None
        try:
            # Создаем временный файл
            temp_file = f"temp_audio_{uuid.uuid4().hex}.wav"
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            # Транскрибируем через API
            text = await self.transcribe_audio(temp_file)
            
            if text:
                response = await self.bot.ai.handle_audio_transcription(text)
                if response:
                    await self.bot.connected_channels[0].send(
                        f"[Аудио-ответ] {response}"
                    )
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
        finally:
            # Удаляем временный файл
            if temp_file and Path(temp_file).exists():
                Path(temp_file).unlink()

    async def transcribe_audio(self, audio_path: str) -> Optional[str]:
        base64_audio = await self.encode_audio_base64(audio_path)
        if not base64_audio:
            return None

        audio_format = audio_path.split('.')[-1].lower()
        payload = {
            "model": "openai-audio",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this audio"},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_audio,
                                "format": audio_format
                            }
                        }
                    ]
                }
            ]
        }

        try:
            async with self.session.post(
                "https://text.pollinations.ai/openai",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('choices', [{}])[0].get('message', {}).get('content')
        
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {str(e)}")
            return None

    async def encode_audio_base64(self, audio_path: str) -> Optional[str]:
        try:
            async with aiohttp.streams.FileSender() as fs:
                content = await fs.file_send(audio_path)
                return base64.b64encode(content).decode('utf-8')
        except Exception as e:
            logger.error(f"Audio encoding error: {str(e)}")
            return None

    async def close(self):
        await self.session.close()

class ImageGenerator:
    @staticmethod
    def _generate_seed() -> int:
        return random.randint(0, 99999)

    async def generate_image(self, prompt: str) -> str:
        try:
            encoded_prompt = urllib.parse.quote(prompt)
            url = f"{API_CONFIG['image_base_url']}/prompt/{encoded_prompt}"
            
            params = {
                "width": API_CONFIG['image_settings']['width'],
                "height": API_CONFIG['image_settings']['height'],
                "model": API_CONFIG['image_settings']['model'],
                "nologo": API_CONFIG['image_settings']['nologo'],
                "enhance": API_CONFIG['image_settings']['enhance'],
                "seed": self._generate_seed()
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    response.raise_for_status()
                    logger.info(f"Generated image: {str(response.url)}")
                    return str(response.url)
            
        except Exception as e:
            logger.error(f"Image generation error: {str(e)}")
            return None

class ImageAnalyzer:
    @staticmethod
    def is_valid_url(url: str) -> bool:
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    async def analyze_image(self, image_url: str, question: str) -> str:
        try:
            if not self.is_valid_url(image_url):
                return "Неверный URL изображения"

            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=15) as response:
                    if response.status != 200:
                        return "Ошибка загрузки изображения"
                    
                    content = await response.read()
                    if len(content) > API_CONFIG['max_image_size']:
                        return "Изображение слишком большое (макс. 4MB)"

                    content_type = response.headers.get('Content-Type', '')
                    if 'image/' not in content_type:
                        return "Неподдерживаемый тип содержимого"
                    
                    image_format = content_type.split('/')[-1].lower()
                    logger.info(f"Detected image format: {image_format}")
                    
                    if image_format not in API_CONFIG['allowed_formats']:
                        return f"Неподдерживаемый формат: {image_format}"

                    base64_image = base64.b64encode(content).decode('utf-8')
                    
                    payload = {
                        "model": API_CONFIG['vision_model'],
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{image_format};base64,{base64_image}"
                                    }
                                }
                            ]
                        }]
                    }

                    async with session.post(
                        f"{API_CONFIG['text_base_url']}/openai",
                        json=payload,
                        timeout=30
                    ) as api_response:
                        api_response.raise_for_status()
                        data = await api_response.json()
                        return data['choices'][0]['message']['content']

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return "Ошибка анализа"

class TextGenerator:
    async def generate_text(self, username: str, messages: list, model: str = None) -> str:
        model = model or API_CONFIG['default_text_model']
        try:
            if model == "searchgpt":
                return await self._handle_search(messages[-1]['content'])
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 50,
                "temperature": 0.7
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{API_CONFIG['text_base_url']}/openai",
                    json=payload,
                    timeout=30
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return self._format_response(data['choices'][0]['message']['content'], model)
            
        except Exception as e:
            logger.error(f"Text generation error: {str(e)}")
            return None

    async def _handle_search(self, query: str) -> str:
        try:
            modified_query = f"Кратко ответь 1-2 предложениями: {query}"
            encoded_query = urllib.parse.quote(modified_query)
            url = f"{API_CONFIG['text_base_url']}/{encoded_query}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params={"model": "searchgpt"},
                    timeout=25
                ) as response:
                    response.raise_for_status()
                    text = await response.text()
                    return self._format_response(text, "searchgpt")
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return None

    def _format_response(self, response: str, model: str) -> str:
        response = response.strip()
        return response

class AIChatBot:
    def __init__(self):
        self.base_context = [{
            "role": "system",
            "content": "Отвечай кратко (1-2 предложения), используй разговорный стиль. До 500 символов. Если с тобой разговаривают агрессивно, можешь отвечать аналогично."
        }]
        self.user_context = {}
        self.request_queue = asyncio.Queue()
        self.image_generator = ImageGenerator()
        self.text_generator = TextGenerator()
        self.image_analyzer = ImageAnalyzer()
        self.audio_context = deque(maxlen=5)

    async def handle_audio_transcription(self, text: str):
        self.audio_context.append({"role": "user", "content": text})
        
        messages = [{
            "role": "system", 
            "content": "Ты отвечаешь на голосовые сообщения. Будь кратким и естественным."
        }] + list(self.audio_context)
        
        response = await self.text_generator.generate_text("audio_user", messages)
        
        if response:
            self.audio_context.append({"role": "assistant", "content": response})
            return response

    def _prepare_search_context(self, query: str) -> list:
        return [{
            "role": "system",
            "content": "Давай краткие ответы (1-2 предложения) на основе поиска. Не используй маркеры списков."
        }] + self.base_context

    def _prepare_context(self, username: str, prompt: str) -> list:
        base_context = [{
            "role": "system",
            "content": f"Ты общаешься с {username}. Отвечай кратко и ёмко. Текущий запрос: {prompt[:450]}..."
        }]

        if username not in self.user_context:
            self.user_context[username] = deque(maxlen=1000)
        
        self.user_context[username].append({
            "role": "user",
            "content": prompt[:450]
        })
        
        return base_context + list(self.user_context[username])

    async def process_requests(self):
        while True:
            username, command, message, future = await self.request_queue.get()
            try:
                if not isinstance(username, str) or not re.match(r"^[\w]+$", username):
                    logger.error(f"Invalid username: {username}")
                    future.set_result("Ошибка: некорректный пользователь")
                    continue
                
                if username not in self.user_context:
                    self.user_context[username] = deque(maxlen=1000)
                    logger.info(f"Создан новый контекст для {username}")

                if command == "analyze":
                    if ' ' not in message:
                        future.set_result("Формат: !analyze [URL] [Вопрос]")
                        continue
                    
                    image_url, question = message.split(' ', 1)
                    response = await self.image_analyzer.analyze_image(image_url, question)
                    future.set_result(response)
                
                elif command == "image":
                    response = await self.image_generator.generate_image(message)
                    future.set_result(response)
                
                elif command == "search":
                    if not message.strip():
                        future.set_result("Укажите запрос!")
                        continue
                    
                    self.user_context[username].extend([
                        {"role": "user", "content": f"Поиск: {message}"},
                        {"role": "assistant", "content": "Ищу информацию..."}
                    ])
                    
                    response = await self.text_generator.generate_text(
                        username, 
                        [{"role": "user", "content": message}],
                        "searchgpt"
                    )
                    future.set_result(response)

                else:  # Обработка обычных текстовых запросов
                    if not message.strip():
                        future.set_result("Напишите запрос после '!'")
                        continue
                        
                    messages = self._prepare_context(username, message)
                    response = await self.text_generator.generate_text(username, messages)
                    if response:
                        self.user_context[username].append({
                            "role": "assistant", 
                            "content": response
                        })
                    future.set_result(response)

            except Exception as e:
                logger.error(f"Processing error: {str(e)}", exc_info=True)
                if future and not future.done():
                    future.set_result(None)
            finally:
                self.request_queue.task_done()

class Bot(commands.Bot):
    def __init__(self, config):
        super().__init__(
            token=config['oauth_token'],
            client_id=config['client_id'],
            initial_channels=[config['channel']],
            prefix='!',
            case_insensitive=True
        )
        self.ai = AIChatBot()
        self.request_tasks = {}
        self.audio_processor = AudioProcessor(self)
    
    

    async def handle_listen(self, username: str, text: str, message):
        try:
            if text.lower().startswith("start"):
                channel = text[5:].strip() or API_CONFIG['channel']
                await self.audio_processor.start_listening(channel)
                await message.channel.send(f"@{username} Начинаю прослушивание канала {channel}!")
            
            elif text.lower() == "stop":
                await self.audio_processor.stop_listening()
                await message.channel.send(f"@{username} Прослушивание остановлено!")
            
            else:
                await message.channel.send(f"@{username} Используйте: !listen [start|stop] [channel]")
        except Exception as e:
            logger.error(f"Listen command error: {str(e)}")
            await message.channel.send(f"@{username} Ошибка: {str(e)}")

    async def event_ready(self):
        logger.info(f"Бот {self.nick} запущен")
        asyncio.create_task(self.ai.process_requests())

    async def event_message(self, message):
        if message.echo or message.author.name == self.nick:
            return
        
        if message.content.startswith('!'):
            await self.handle_command(message)

    async def handle_command(self, message):
        raw_content = message.content.lstrip('!').strip()
        username = message.author.name

        if not raw_content:
            await message.channel.send(f"@{username} Введите команду после '!'")
            return

        command_handlers = {
            'help': self.handle_help,
            'image': self.handle_image,
            'analyze': self.handle_analyze,
            'search': self.handle_search,
            'listen': self.handle_listen  # ⚠️ Добавлен обработчик для listen
        }

        for cmd in command_handlers:
            if raw_content.startswith(cmd):
                handler = command_handlers[cmd]
                text = raw_content[len(cmd):].strip()
                await handler(username, text, message)
                return

        # Если команда не распознана - обрабатываем как текстовый запрос
        await self.handle_chat(username, raw_content, message)

    async def handle_help(self, username: str, text: str, message):
        help_text = (
            "Команды: | "
            "!help - помощь | "
            "!image [описание] - генерация изображения | "
            "!analyze [URL] [вопрос] - анализ изображения | "
            "!search [запрос] - поиск информации | "
            "Любой другой текст после '!' - общение с нейросетью"
        )
        await message.channel.send(f"@{username} {help_text}")

    async def handle_image(self, username: str, text: str, message):
        if not text:
            await message.channel.send(f"@{username} Укажите описание изображения!")
            return

        await self.process_request(username, 'image', text, message)

    async def handle_analyze(self, username: str, text: str, message):
        if not text:
            await message.channel.send(f"@{username} Формат: !analyze [URL] [Вопрос]")
            return

        await self.process_request(username, 'analyze', text, message)

    async def handle_search(self, username: str, text: str, message):
        if not text:
            await message.channel.send(f"@{username} Укажите поисковый запрос!")
            return

        await self.process_request(username, 'search', text, message)

    async def handle_chat(self, username: str, text: str, message):
        await self.process_request(username, 'chat', text, message)

    def split_message(self, text: str, max_length: int = 450) -> list:
        parts = []
        while len(text) > 0:
            if len(text) <= max_length:
                parts.append(text)
                break
            
            ideal_split = max_length
            split_pos = text.rfind(' ', 0, ideal_split)
        
            if split_pos == -1:
                split_pos = text.find(' ', ideal_split)
                if split_pos == -1:
                    split_pos = min(len(text), max_length)
        
            part = text[:split_pos].strip()
            if part:
                parts.append(part)
            text = text[split_pos:].strip()
    
        optimized = []
        for part in parts:
            if optimized and len(optimized[-1]) + len(part) + 1 <= max_length:
                optimized[-1] += " " + part
            else:
                optimized.append(part)
    
        return optimized

    async def process_request(self, username: str, command: str, text: str, message):
        try:
            future = asyncio.Future()
            request_id = uuid.uuid4()
            self.request_tasks[request_id] = future
            
            await self.ai.request_queue.put((username, command, text, future))
            
            response = await asyncio.wait_for(future, timeout=45)
            
            if response:
                parts = self.split_message(response)
                for i, part in enumerate(parts):
                    if len(parts) > 1:
                        part = f"({i+1}/{len(parts)}) {part}"
                    await message.channel.send(f"@{username} {part}")
                    if i < len(parts)-1:
                        await asyncio.sleep(1)

        except asyncio.TimeoutError:
            await message.channel.send(f"@{username} Таймаут выполнения")
        except Exception as e:
            logger.error(f"Ошибка: {str(e)}")
            await message.channel.send(f"@{username} Произошла ошибка")
        finally:
            self.request_tasks.pop(request_id, None)

def load_config():
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

async def main():
    config = load_config()
    bot = Bot(config)
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())