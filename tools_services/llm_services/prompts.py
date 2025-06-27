class LLMPrompts:
    """Class chứa các prompt templates cho các tác vụ khác nhau"""
    
    @staticmethod
    def query_augmentation_prompt(original_query: str) -> str:
        """Prompt cho tác vụ query augmentation"""
        return f"""Bạn là một chuyên gia y học. Từ câu hỏi gốc của người dùng, hãy tạo ra:
1. Một Structured Query: Mô tả lại câu hỏi theo cấu trúc rõ ràng (thuốc gì, bệnh gì, gene gì)
2. Tối đa 3 Augmented Queries: Các câu hỏi suy luận cần thiết để trả lời câu hỏi gốc

Câu hỏi gốc: {original_query}

Trả lời theo định dạng JSON:
{{
    "structured_query": "Structured query ở đây",
    "augmented_queries": ["Aug query 1", "Aug query 2", "Aug query 3"]
}}"""
    
    @staticmethod
    def summary_web_results_prompt(aug_query: str, web_content: str) -> str:
        """Prompt cho tác vụ tóm tắt kết quả web search"""
        return f"""Bạn là một chuyên gia y học. Hãy tóm tắt nội dung web sau để trả lời câu hỏi cụ thể.

Câu hỏi cần trả lời: {aug_query}

Nội dung web:
{web_content}

Hãy tóm tắt các thông tin quan trọng và liên quan trực tiếp đến câu hỏi. Chỉ giữ lại thông tin chính xác và đáng tin cậy. Trả lời bằng tiếng Việt, ngắn gọn và súc tích (tối đa 200 từ)."""
    
    @staticmethod
    def reflection_prompt(structured_query: str, aug_queries: list, context: str) -> str:
        """Prompt cho tác vụ reflection"""
        aug_queries_str = "\n".join([f"- {q}" for q in aug_queries])
        
        return f"""Bạn là một chuyên gia đánh giá thông tin y học. Hãy xem xét xem thông tin hiện có đã đủ để trả lời các câu hỏi sau hay chưa.

Câu hỏi chính: {structured_query}

Các câu hỏi phụ:
{aug_queries_str}

Thông tin hiện có:
{context}

Hãy đánh giá và trả lời theo định dạng JSON:
{{
    "sufficient": true/false,
    "reasoning": "Lý do tại sao đủ hoặc không đủ thông tin",
    "follow_up_queries": ["Câu hỏi bổ sung 1", "Câu hỏi bổ sung 2"] hoặc []
}}

Chỉ trả về JSON, không thêm text khác."""
    
    @staticmethod
    def answer_prompt(original_query: str, context: str, chat_history: list) -> str:
        """Prompt cho tác vụ trả lời cuối cùng"""
        history_str = ""
        if chat_history:
            for item in chat_history[-3:]:  # Chỉ lấy 3 cuộc hội thoại gần nhất
                history_str += f"Người dùng: {item.get('query', '')}\nTrợ lý: {item.get('answer', '')}\n\n"
        
        return f"""Bạn là một trợ lý AI chuyên về y học và dược học. Hãy trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp.

Lịch sử cuộc hội thoại:
{history_str}

Câu hỏi hiện tại: {original_query}

Thông tin tham khảo:
{context}

Hướng dẫn trả lời:
- Trả lời chính xác, dựa trên thông tin được cung cấp
- Sử dụng tiếng Việt tự nhiên, dễ hiểu
- Cấu trúc rõ ràng với đầu mục nếu cần
- Đề cập nguồn thông tin nếu có
- Nếu thông tin không đủ, hãy nói rõ và đưa ra lời khuyên chung
- Luôn khuyến cáo tham khảo ý kiến bác sĩ cho các vấn đề y tế

Câu trả lời:"""
    
    @staticmethod
    def general_prompt(query: str) -> str:
        """Prompt cho câu hỏi không liên quan đến y học"""
        return f"""Câu hỏi của bạn: {query}

Tôi là trợ lý AI chuyên về y học và dược học. Câu hỏi của bạn có vẻ không liên quan đến lĩnh vực chuyên môn của tôi.

Tôi có thể giúp bạn trả lời các câu hỏi về:
- Thông tin về thuốc (công dụng, tác dụng phụ, liều dùng)
- Tương tác thuốc với thuốc khác
- Tương tác thuốc với gene
- Mối liên hệ giữa thuốc và bệnh
- Thông tin về các bệnh lý
- Tương tác gen-bệnh

Bạn có muốn hỏi về những chủ đề này không?"""
