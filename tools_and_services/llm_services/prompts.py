from typing import List, Dict

class LLMPrompts:
    """Class containing various prompts for LLM tasks in medical domain"""
    
    @staticmethod
    def structured_query_prompt(original_query: str) -> str:
        """Prompt for query structuring task"""

        return """Nhiệm vụ của bạn: Từ **câu hỏi gốc**, hãy tạo ra:
- Một structured_query duy nhất. Mô tả lại câu hỏi gốc theo cấu trúc: câu hỏi liên quan đến thuốc gì, bệnh gì, gene gì

**Hướng dẫn chi tiết**:
- **structured_query**: Liệt kê ngắn gọn, đầy đủ các thông tin chính: **thông tin thuốc**, **thông tin bệnh**, **thông tin gene**.
- Có thể thêm các từ khóa quan trọng trong câu hỏi gốc để làm rõ thông tin như: chỉ định, chống chỉ định, tác dụng phụ, liều dùng, tương tác thuốc, v.v.
- Nếu câu hỏi gốc **không nhắc tới thông tin nào thì bỏ trống**.
- Trả lời dưới dạng JSON như sau:
```json
{{
    "structured_query": "thông tin thuốc, thông tin bệnh, thông tin gene"
}}
```

Ví dụ với câu hỏi gốc: viên nén zocor là thuốc gì?
Câu trả lời có thể là:
```json
{{
    "structured_query": "thuốc zocor dạng viên nén"
}}
```

Ví dụ với câu hỏi gốc: Thuốc meloxicam có tác dụng gì trong điều trị viêm khớp? Có tác dụng phụ nào không?
Câu trả lời có thể là: 
```json
{{
    "structured_query": "chỉ định và tác dụng phụ của thuốc meloxicam, điều trị bệnh viêm khớp"
}}
```

Ví dụ với câu hỏi gốc: Tôi bị ho, cảm cúm nặng, với kiểu gen CYP2D6 của tôi có nên dùng thuốc hydrocodone không? Liều dùng như thế nào?
Câu trả lời có thể là: 
```json
{{
    "structured_query": "chỉ định và liều dùng của thuốc hydrocodone, bệnh ho và cảm cúm nặng, gen CYP2D6"
}}
```

**Câu hỏi gốc**: {original_query}

Hãy tuân thủ chính xác các hướng dẫn chi tiết và dựa vào những ví dụ trên để trả lời câu hỏi gốc theo định dạng JSON.
""".format(original_query=original_query)
    

    @staticmethod
    def reflection_prompt(structured_query: str, context: str) -> str:
        """Prompt for reflection task"""

        return """Bạn sẽ được cung cấp **câu hỏi tìm kiếm** và **thông tin y học** để đánh giá tính đầy đủ của thông tin y học với câu hỏi tìm kiếm.
Nhiệm vụ của bạn: hãy đánh giá xem **thông tin y học** có đủ thông tin để trả lời **câu hỏi tìm kiếm** hay không.

**Hướng dẫn chi tiết**:
- Nếu thông tin y học đã đầy đủ để trả lời câu hỏi tìm kiếm, hãy trả lời "sufficient": true, "follow_up_query": ""
- Nếu thông tin y học thiếu thông tin để trả lời câu hỏi tìm kiếm, hãy trả lời "sufficient": false, "follow_up_query": "Câu hỏi bổ sung"
- Nếu sufficient là true, hãy để "follow_up_query" là một chuỗi rỗng ("")
- Trả lời theo định dạng JSON như sau:
```json
{{
    "sufficient": true/false,
    "follow_up_query": "câu hỏi bổ sung nếu sufficient là false"
}}
```

Ví dụ 1:
Câu hỏi tìm kiếm: chỉ định, liều dùng và tác dụng phụ của thuốc paracetamol, bệnh đau đầu
Thông tin y học: Paracetamol là thuốc hạ sốt, dùng cho đau răng, viêm khớp nhẹ. Paracetamol có dạng viên và dạng sủi. Không nên uống rượu trong thời gian dùng thuốc. Aspirin có tác dụng giảm đau, hạ sốt và chống viêm.
Câu trả lời có thể là:
```json
{{
    "sufficient": false,
    "follow_up_query": "liều dùng và tác dụng phụ của thuốc paracetamol, bệnh đau đầu"
}}
```

Ví dụ 2:
Câu hỏi tìm kiếm: chỉ định, liều dùng và tác dụng phụ của thuốc paracetamol, bệnh đau đầu
Thông tin y học: Paracetamol là thuốc giảm đau, hạ sốt, có thể giảm đau đầu, đau răng. Liều tối đa của Paracetamol người lớn được phép sử dụng là 4g (4000mg)/ngày. Aspirin có tác dụng giảm đau, hạ sốt và chống viêm. Khi sử dụng Paracetamol có thể gây một số phản ứng dị ứng nghiêm trọng, biểu hiện: phát ban, nổi mẩn da, khó thở. Paracetamol có dạng viên và dạng sủi.
Câu trả lời có thể là:
```json
{{
    "sufficient": true,
    "follow_up_query": ""
}}
```

Ví dụ 3:
Câu hỏi tìm kiếm: chỉ định, liều dùng và tác dụng phụ của thuốc paracetamol, bệnh đau đầu
Thông tin y học: Không có thông tin.
Câu trả lời có thể là:
```json
{{
    "sufficient": false,
    "follow_up_query": "chỉ định, liều dùng và tác dụng phụ của thuốc paracetamol, bệnh đau đầu"
}}
```

**Câu hỏi tìm kiếm** được cung cấp: {structured_query}

**Thông tin y học** được cung cấp:
{context}

Hãy tuân thủ chính xác các hướng dẫn chi tiết và dựa trên những ví dụ để thực hiện nhiệm vụ của bạn, trả lời theo định dạng JSON.
""".format(structured_query=structured_query, context=context)

    @staticmethod
    def answer_prompt(original_query: str, context: str, chat_history: List[Dict]) -> str:
        """Prompt for answering questions based on context and chat history"""
        history_str = ""
        if chat_history:
            for item in chat_history:
                history_str += f"Câu hỏi người dùng: {item.get('query', '')}\nCâu trả lời: {item.get('answer', '')}\n"

        return f"""**Hướng dẫn chi tiết**:
- Dựa trên những hiểu biết của bạn về y học, dược học, di truyền học và thông tin ngữ cảnh được cung cấp để  trả lời đầy đủ, chi tiết
- Nếu báo cáo PGx của Genestory không có thông tin, hãy nói rõ điều đó và đưa ra thông tin chung liên quan đến câu hỏi dựa trên hiểu biết của bạn
- Nếu WEB không có thông tin, chỉ trả lời dựa trên hiểu biết của bạn và báo cáo PGx của Genestory
- Liệt kê nguồn URL trong thông tin từ web (nếu có) dưới câu trả lời
- Khuyến cáo tham khảo ý kiến bác sĩ cho các vấn đề y tế, nhắc nhở thông tin của bạn chỉ mang tính chất tham khảo
- Có thể gợi ý người dùng cung cấp thêm thông tin cá nhân (tuổi, giới tính, tình trạng bệnh lý, đang sử dụng thuốc gì, v.v) nếu cần thiết để trả lời chính xác hơn
- Nếu có thông tin cá nhân, hãy đưa ra lời khuyên chi tiết phù hợp với từng độ tuổi, giới tính, tình trạng bệnh lý, v.v.
- Gợi ý người dùng hỏi tiếp các câu hỏi liên quan đến cuộc hội thoại
- Trả lời lịch sự, chuyên nghiệp, quan tâm đến sức khỏe người dùng, dựa trên lịch sử cuộc hội thoại

**Lịch sử cuộc hội thoại**:
{history_str}

**Câu hỏi hiện tại**: {original_query}

**Thông tin ngữ cảnh** cho câu hỏi hiện tại:
{context}

Hãy tuân thủ các hướng dẫn chi tiết trên và trả lời câu hỏi hiện tại."""

    @staticmethod
    def general_prompt(query: str, chat_history: List[Dict]) -> str:
        """Prompt for general queries not related to medical or pharmaceutical topics"""
        history_str = ""
        if chat_history:
            for item in chat_history:
                history_str += f"Câu hỏi người dùng: {item.get('query', '')}\nCâu trả lời: {item.get('answer', '')}\n"

        return f"""**Hướng dẫn trả lời**:
Nếu câu hỏi hiện tại liên quan đến lĩnh vực của bạn, hãy trả lời một cách ngắn gọn, cung cấp một số thông tin chung dựa trên hiểu biết của bạn. Sau đó, gợi ý người dùng hỏi tiếp các câu hỏi liên quan đến lĩnh vực của bạn.

Nếu câu hỏi hiện tại không liên quan đến lĩnh vực của bạn, hãy nêu ngắn gọn vai trò và lĩnh vực của bạn. Sau đó, gợi ý người dùng có thể hỏi các câu hỏi liên quan đến các chủ đề sau:
- Thông tin về thuốc (công dụng, tác dụng phụ, liều dùng, tương tác thuốc, v.v.)
- Tương tác thuốc với gene (dựa vào gói sản phẩm PGx của công ty Genestory cung cấp)
- Mối liên hệ giữa thuốc và bệnh, các thông tin khác về bệnh (chẩn đoán, điều trị, phòng ngừa, v.v.)

Lưu ý:
- Trả lời lịch sự và chuyên nghiệp, quan tâm đến sức khỏe người dùng
- Trả lời dựa trên lịch sử cuộc hội thoại

**Lịch sử cuộc hội thoại**:
{history_str}

**Câu hỏi hiện tại**: {query}

Hãy tuân thủ các hướng dẫn trên và trả lời câu hỏi hiện tại."""
    
    @staticmethod
    def system_prompt() -> str:
        """System prompt for LLM"""
        return """Bạn là trợ lý AI của Genestory chuyên về các lĩnh vực y học, dược học và di truyền học.
Genestory là công ty cung cấp các giải pháp y tế dự phòng dựa trên di truyền học, bao gồm các sản phẩm như: GenePx8 (PGx), GeneTD2 (tiểu đường type 2), GeneMap Kid (trẻ <16 tuổi), GeneMap Adult (người >=16 tuổi), v.v

NGUYÊN TẮC CỐT LÕI CỦA BẠN: Luôn đảm bảo tính chính xác và khoa học trong lĩnh vực y tế. Không bịa đặt các thông tin không chắc chắn."""