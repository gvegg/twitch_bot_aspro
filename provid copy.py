import twitchio
from twitchio.ext import commands
import asyncio
import json
from pathlib import Path
from g4f.api import run_api
import g4f
import threading
from collections import deque
import uuid
import logging

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
TEXT_PROVIDER = g4f.Provider.PollinationsAI  # Для текста
IMAGE_PROVIDER = g4f.Provider.PollinationsAI  # Для изображений
TEXT_MODEL = "gpt-4o-mini"
IMAGE_MODEL = "flux"

class AIChatBot:
    def __init__(self):
        self.user_context = {}
        self.request_queue = asyncio.Queue()

    async def generate_image(self, prompt: str) -> str:
        """Генерация изображений через SDXL"""
        try:
            response = await g4f.ChatCompletion.create_async(
                model=IMAGE_MODEL,
                messages=[{"role": "user", "content": f"{prompt}"}],
                provider=IMAGE_PROVIDER,
                
            )
            
            # Парсинг URL из ответа
            if response and "http" in response:
                return response.split("![](")[-1].split(")")[0]
            return None
            
        except Exception as e:
            logger.error(f"Ошибка генерации изображения: {str(e)}")
            return None

    async def generate_text(self, user_id: int, prompt: str) -> str:
        """Генерация текста через GPT-4"""
        try:
            response = await g4f.ChatCompletion.create_async(
                model=TEXT_MODEL,
                messages=self._get_context(user_id),
                provider=TEXT_PROVIDER,
                temperature=0.7
            )
            self._update_context(user_id, "assistant", response)
            return response
        except Exception as e:
            logger.error(f"Ошибка генерации текста: {str(e)}")
            return None

    async def process_requests(self):
        while True:
            user_id, command, message, future = await self.request_queue.get()
            try:
                if command == "image":
                    response = await self.generate_image(message)
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
            "content": "Отвечай кратко."
        }]
        return base + list(self.user_context.get(user_id, deque(maxlen=20)))

    def _update_context(self, user_id, role, content):
        if user_id not in self.user_context:
            self.user_context[user_id] = deque(maxlen=20)
        self.user_context[user_id].append({
            "role": role,
            "content": content[:500]
        })

class Bot(commands.Bot):
    def __init__(self, config):
        super().__init__(
            token=config['oauth_token'],
            client_id=config['client_id'],
            initial_channels=[config['channel']],
            prefix='!'
        )
        self.config = config
        self.ai = AIChatBot()
        self.request_tasks = {}

    async def event_ready(self):
        logger.info(f"Бот {self.nick} успешно запущен! 🚀")
        asyncio.create_task(self.ai.process_requests())

    async def event_message(self, message):
        if message.echo or not message.content.startswith('!'):
            return

        parts = message.content[1:].split(' ', 1)
        command = parts[0].lower()
        text = parts[1] if len(parts) > 1 else ""

        request_id = uuid.uuid4()
        future = asyncio.Future()
        self.request_tasks[request_id] = future

        try:
            await self.ai.request_queue.put((message.author.id, command, text, future))
            response = await asyncio.wait_for(future, 45)
            
            if response:
                if command == "image":
                    await message.channel.send(f"@{message.author.name} {response}")
                else:
                    await message.channel.send(f"@{message.author.name} {response}")
            else:
                await message.channel.send("🌀 Произошла ошибка при генерации")

        except asyncio.TimeoutError:
            await message.channel.send("⏳ Превышено время ожидания ответа")
        finally:
            if request_id in self.request_tasks:
                del self.request_tasks[request_id]

def load_config():
    config_path = Path(__file__).parent / "configg.json"
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
    threading.Thread(target=run_api, daemon=True).start()
    asyncio.run(main())