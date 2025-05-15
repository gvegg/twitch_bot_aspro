import base64  # NEW: Добавлен импорт
import re  # NEW: Для проверки URL

# ... (остальные импорты без изменений)

# Конфигурация API
API_CONFIG = {
    # ... (старые настройки)
    "max_image_size": 4 * 1024 * 1024,  # NEW: 4MB лимит
    "allowed_formats": ["jpg", "jpeg", "png", "webp"],  # NEW
    "vision_model": "openai-vision"  # NEW
}

# NEW: Добавлен класс для анализа изображений
class ImageAnalyzer:
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Проверка валидности URL изображения"""
        regex = re.compile(
            r'^(?:http)s?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return re.match(regex, url) is not None

    async def analyze_image(self, image_url: str, question: str) -> str:
        try:
            # Проверка URL
            if not self.is_valid_url(image_url):
                return "Неверный URL изображения"

            # Загрузка изображения
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(image_url, timeout=15)
            )

            if response.status_code != 200:
                return "Не удалось загрузить изображение"
            
            if len(response.content) > API_CONFIG['max_image_size']:
                return "Изображение слишком большое (макс. 4MB)"

            # Проверка формата
            image_format = image_url.split('.')[-1].lower()
            if image_format not in API_CONFIG['allowed_formats']:
                return "Неподдерживаемый формат изображения"

            # Подготовка запроса
            base64_image = base64.b64encode(response.content).decode('utf-8')
            
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

            # Отправка запроса
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{API_CONFIG['text_base_url']}/openai",
                    json=payload,
                    timeout=30
                )
            )

            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content'][:400]

        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}")
            return "Ошибка анализа изображения"

class AIChatBot:
    def __init__(self):
        self.user_context = {}
        self.request_queue = asyncio.Queue()
        self.image_generator = ImageGenerator()
        self.text_generator = TextGenerator()
        self.image_analyzer = ImageAnalyzer()  # NEW

    async def process_requests(self):
        while True:
            user_id, command, message, future = await self.request_queue.get()
            try:
                # NEW: Добавлена обработка команды analyze
                if command == "analyze":
                    parts = message.split(' ', 1)
                    if len(parts) < 2:
                        future.set_result("Формат: !analyze [URL] [Вопрос]")
                        continue
                    
                    image_url, question = parts[0], parts[1]
                    response = await self.image_analyzer.analyze_image(image_url, question)
                    future.set_result(response)
                elif command == "image":
                    response = await self.image_generator.generate_image(message)
                elif command == "search":
                    # ... (старая логика без изменений)

class Bot(commands.Bot):
    async def handle_command(self, message):
        content = message.content.lstrip('!')
        parts = content.split(' ', 1)
        command = parts[0].lower()
        text = parts[1] if len(parts) > 1 else ""

        # NEW: Обработка команды analyze
        if command == "analyze":
            if not text:
                await message.channel.send(f"@{message.author.name} Формат: !analyze [URL изображения] [Вопрос]")
                return

            request_id = uuid.uuid4()
            future = asyncio.Future()
            self.request_tasks[request_id] = future

            try:
                await self.ai.request_queue.put((message.author.id, "analyze", text, future))
                response = await asyncio.wait_for(future, 45)

                if response:
                    await message.channel.send(f"@{message.author.name} 📷 Анализ: {response}")
                else:
                    await message.channel.send(f"@{message.author.name} Ошибка анализа")

            except asyncio.TimeoutError:
                await message.channel.send(f"@{message.author.name} Таймаут анализа")
            finally:
                self.request_tasks.pop(request_id, None)
            return

        # ... (старая логика обработки других команд)

# ... (остальной код без изменений)