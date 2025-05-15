import twitchio
from twitchio.ext import commands
import asyncio
import json
from pathlib import Path
from collections import deque
import uuid
import logging
import requests
import urllib.parse
import random
import g4f

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TwitchBot")

# Конфигурация
TEXT_PROVIDER = g4f.Provider.PollinationsAI
TEXT_MODEL = "gpt-4o-mini"
IMAGE_SETTINGS = {
    "width": 3840,
    "height": 2160,
    "model": "flux",
    "nologo": "true"
}

class ImageGenerator:
    @staticmethod
    def _generate_seed() -> int:
        return random.randint(0, 9999)

class ImageGenerator:
    @staticmethod
    def _generate_seed() -> int:
        return random.randint(0, 9999)

    async def generate_image(self, prompt: str) -> str:
        """Генерация изображений через Pollinations API"""
        try:
            encoded_prompt = urllib.parse.quote(prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            
            params = {
                **IMAGE_SETTINGS,
                "seed": self._generate_seed()
            }

            # Исправленный блок выполнения запроса
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,  # Используем дефолтный executor
                lambda: requests.get(
                    url, 
                    params=params, 
                    timeout=30
                )
            )
            
            response.raise_for_status()
            return response.url
            
        except Exception as e:
            logger.error(f"Ошибка генерации изображения: {str(e)}")
            return None

class AIChatBot:
    def __init__(self):
        self.user_context = {}
        self.request_queue = asyncio.Queue()
        self.image_generator = ImageGenerator()

    async def generate_text(self, user_id: int, prompt: str) -> str:
        try:
            response = await g4f.ChatCompletion.create_async(
                model=TEXT_MODEL,
                messages=self._get_context(user_id),
                provider=TEXT_PROVIDER,
                temperature=0.7
            )
            self._update_context(user_id, "assistant", response)
            return response[:400]  # Обрезаем длинные ответы
        except Exception as e:
            logger.error(f"Ошибка генерации текста: {str(e)}")
            return None

    async def process_requests(self):
        while True:
            user_id, command, message, future = await self.request_queue.get()
            try:
                if command == "image":
                    response = await self.image_generator.generate_image(message)
                else:
                    self._update_context(user_id, "user", f"{command} {message}")
                    response = await self.generate_text(user_id, message)
                
                future.set_result(response)
            except Exception as e:
                logger.error(f"Ошибка обработки: {str(e)}")
                future.set_result(None)
            finally:
                self.request_queue.task_done()

    def _get_context(self, user_id):
        base = [{
            "role": "system", 
            "content": "Ты ассистент стримера. Отвечай кратко (1-2 предложения), используй разговорный стиль."
        }]
        return base + list(self.user_context.get(user_id, deque(maxlen=5)))

    def _update_context(self, user_id, role, content):
        if user_id not in self.user_context:
            self.user_context[user_id] = deque(maxlen=5)
        self.user_context[user_id].append({
            "role": role,
            "content": content[:150]
        })

class Bot(commands.Bot):
    def __init__(self, config):
        super().__init__(
            token=config['oauth_token'],
            client_id=config['client_id'],
            initial_channels=[config['channel']],
            prefix='!',
            case_insensitive=True,
            ignore_unregistered_commands=True
        )
        self.config = config
        self.ai = AIChatBot()
        self.request_tasks = {}

    async def event_ready(self):
        logger.info(f"Бот {self.nick} успешно запущен")
        asyncio.create_task(self.ai.process_requests())

    async def event_message(self, message):
        if message.echo:
            return

        if message.content.startswith('!'):
            await self.handle_command(message)

    async def handle_command(self, message):
        content = message.content.lstrip('!')
        command, *args = content.split(' ', 1)
        command = command.lower()
        text = args[0] if args else ""

        logger.info(f"Получена команда: {command} | Текст: {text[:20]}...")

        request_id = uuid.uuid4()
        future = asyncio.Future()
        self.request_tasks[request_id] = future

        try:
            await self.ai.request_queue.put((message.author.id, command, text, future))
            response = await asyncio.wait_for(future, 45)

            if response:
                if command == "image":
                    msg = f"@{message.author.name} {response}"
                else:
                    msg = f"@{message.author.name} {response}"
                
                sent_msg = await message.channel.send(msg)
                
            else:
                await message.channel.send(f"@{message.author.name} Ошибка обработки запроса")

        except asyncio.TimeoutError:
            error_msg = await message.channel.send(f"@{message.author.name} Серверы перегружены")
            await asyncio.sleep(10)
            await error_msg.delete()
        
        except Exception as e:
            logger.error(f"Ошибка обработки команды: {str(e)}")
        
        finally:
            if request_id in self.request_tasks:
                del self.request_tasks[request_id]

def load_config():
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.critical(f"Ошибка загрузки конфига: {str(e)}")
        raise

async def main():
    config = load_config()
    bot = Bot(config)
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())