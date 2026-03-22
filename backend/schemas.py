from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户消息内容")
    user_id: Optional[str] = Field(default="default_user", description="用户ID")
    session_id: Optional[str] = Field(default="default_session", description="会话ID")


class RetrievedChunk(BaseModel):
    """检索到的文档块"""
    filename: str = Field(..., description="来源文件名")
    page_number: Optional[str | int] = Field(default=None, description="页码")
    text: Optional[str] = Field(default=None, description="文档块文本内容")
    score: Optional[float] = Field(default=None, description="相似度分数")
    rrf_rank: Optional[int] = Field(default=None, description="RRF排名")
    rerank_score: Optional[float] = Field(default=None, description="重排序分数")


class RagTrace(BaseModel):
    """RAG检索追踪信息"""
    tool_used: bool = Field(..., description="是否使用了RAG工具")
    tool_name: str = Field(..., description="工具名称")
    query: Optional[str] = Field(default=None, description="原始查询")
    expanded_query: Optional[str] = Field(default=None, description="扩展查询")
    step_back_question: Optional[str] = Field(default=None, description="回退问题")
    step_back_answer: Optional[str] = Field(default=None, description="回退答案")
    expansion_type: Optional[str] = Field(default=None, description="扩展类型")
    hypothetical_doc: Optional[str] = Field(default=None, description="假设文档")
    retrieval_stage: Optional[str] = Field(default=None, description="检索阶段")
    grade_score: Optional[str] = Field(default=None, description="评分分数")
    grade_route: Optional[str] = Field(default=None, description="评分路由")
    rewrite_needed: Optional[bool] = Field(default=None, description="是否需要重写")
    rewrite_strategy: Optional[str] = Field(default=None, description="重写策略")
    rewrite_query: Optional[str] = Field(default=None, description="重写后的查询")
    rerank_enabled: Optional[bool] = Field(default=None, description="是否启用重排序")
    rerank_applied: Optional[bool] = Field(default=None, description="是否应用重排序")
    rerank_model: Optional[str] = Field(default=None, description="重排序模型")
    rerank_endpoint: Optional[str] = Field(default=None, description="重排序端点")
    rerank_error: Optional[str] = Field(default=None, description="重排序错误")
    retrieval_mode: Optional[str] = Field(default=None, description="检索模式")
    candidate_k: Optional[int] = Field(default=None, description="候选数量K")
    leaf_retrieve_level: Optional[int] = Field(default=None, description="叶子检索层级")
    auto_merge_enabled: Optional[bool] = Field(default=None, description="是否启用自动合并")
    auto_merge_applied: Optional[bool] = Field(default=None, description="是否应用自动合并")
    auto_merge_threshold: Optional[int] = Field(default=None, description="自动合并阈值")
    auto_merge_replaced_chunks: Optional[int] = Field(default=None, description="自动合并替换的块数")
    auto_merge_steps: Optional[int] = Field(default=None, description="自动合并步骤数")
    retrieved_chunks: Optional[List[RetrievedChunk]] = Field(default=None, description="检索到的文档块列表")
    initial_retrieved_chunks: Optional[List[RetrievedChunk]] = Field(default=None, description="初始检索块列表")
    expanded_retrieved_chunks: Optional[List[RetrievedChunk]] = Field(default=None, description="扩展检索块列表")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str = Field(..., description="AI回复内容")
    rag_trace: Optional[RagTrace] = Field(default=None, description="RAG追踪信息")


class MessageInfo(BaseModel):
    """消息信息模型"""
    type: str = Field(..., description="消息类型: user/assistant/system")
    content: str = Field(..., description="消息内容")
    timestamp: Optional[str] = Field(default=None, description="时间戳")
    rag_trace: Optional[RagTrace] = Field(default=None, description="RAG追踪信息")


class SessionMessagesResponse(BaseModel):
    """会话消息列表响应"""
    messages: List[MessageInfo] = Field(..., description="消息列表")


class SessionInfo(BaseModel):
    """会话信息模型"""
    session_id: str = Field(..., description="会话ID")
    updated_at: str = Field(..., description="最后更新时间")
    message_count: int = Field(..., description="消息数量")


class SessionListResponse(BaseModel):
    """会话列表响应"""
    sessions: List[SessionInfo] = Field(..., description="会话列表")


class SessionDeleteResponse(BaseModel):
    """会话删除响应"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="响应消息")


class DocumentInfo(BaseModel):
    """文档信息模型"""
    filename: str = Field(..., description="文件名")
    file_type: str = Field(..., description="文件类型")
    chunk_count: int = Field(..., description="文档块数量")
    uploaded_at: Optional[str] = Field(default=None, description="上传时间")


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: List[DocumentInfo] = Field(..., description="文档列表")


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    filename: str = Field(..., description="文件名")
    chunks_processed: int = Field(..., description="处理的块数量")
    message: str = Field(..., description="响应消息")


class DocumentDeleteResponse(BaseModel):
    """文档删除响应"""
    filename: str = Field(..., description="文件名")
    chunks_deleted: int = Field(..., description="删除的块数量")
    message: str = Field(..., description="响应消息")
