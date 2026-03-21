from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., description="用户消息")
    user_id: str = Field(default="default_user", description="用户ID")
    session_id: str = Field(default="default_session", description="会话ID")


class ChatResponse(BaseModel):
    """聊天响应"""
    response: Optional[str] = Field(None, description="AI响应内容")
    rag_trace: Optional[dict] = Field(None, description="RAG追踪信息")


class MessageInfo(BaseModel):
    """消息信息"""
    type: str = Field(..., description="消息类型: human/ai/system")
    content: str = Field(..., description="消息内容")
    timestamp: Optional[str] = Field(None, description="时间戳")
    rag_trace: Optional[dict] = Field(None, description="RAG追踪信息")


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str = Field(..., description="会话ID")
    updated_at: Optional[str] = Field(None, description="最后更新时间")
    message_count: int = Field(0, description="消息数量")


class SessionMessagesResponse(BaseModel):
    """会话消息列表响应"""
    messages: list[MessageInfo] = Field(default_factory=list, description="消息列表")


class SessionListResponse(BaseModel):
    """会话列表响应"""
    sessions: list[SessionInfo] = Field(default_factory=list, description="会话列表")


class SessionDeleteResponse(BaseModel):
    """删除会话响应"""
    session_id: str = Field(..., description="被删除的会话ID")
    message: str = Field(..., description="操作结果信息")


class DocumentInfo(BaseModel):
    """文档信息"""
    filename: str = Field(..., description="文件名")
    file_type: str = Field(..., description="文件类型")
    chunk_count: int = Field(0, description="分块数量")


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: list[DocumentInfo] = Field(default_factory=list, description="文档列表")


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    filename: str = Field(..., description="文件名")
    chunks_processed: int = Field(..., description="处理的分块数量")
    message: str = Field(..., description="操作结果信息")


class DocumentDeleteResponse(BaseModel):
    """删除文档响应"""
    filename: str = Field(..., description="被删除的文件名")
    chunks_deleted: int = Field(0, description="删除的分块数量")
    message: str = Field(..., description="操作结果信息")
