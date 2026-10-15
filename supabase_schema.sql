-- ==============================================================================
-- SUPABASE POSTGRESQL + PGVECTOR SCHEMA FOR TELEGRAM AI USERBOT MEMORY
-- ==============================================================================
-- Run this in your Supabase project's SQL Editor (https://supabase.com/dashboard/project/_/sql)

-- 1. Enable the pgvector extension to work with embedding vectors
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the chat_memory table
CREATE TABLE IF NOT EXISTS chat_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_id BIGINT NOT NULL,
    sender_id BIGINT NOT NULL,
    sender_name TEXT,
    is_owner BOOLEAN DEFAULT FALSE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    message_type TEXT NOT NULL DEFAULT 'text' CHECK (message_type IN ('text', 'voice', 'video', 'photo', 'document', 'system')),
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Create standard indexes for rapid query performance
CREATE INDEX IF NOT EXISTS idx_chat_memory_chat_id ON chat_memory(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_memory_created_at ON chat_memory(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_memory_chat_created ON chat_memory(chat_id, created_at DESC);

-- 4. Create an HNSW index on the vector embedding column for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_chat_memory_embedding_hnsw 
ON chat_memory 
USING hnsw (embedding vector_cosine_ops);

-- 5. Create the RPC function for vector similarity search (RAG)
CREATE OR REPLACE FUNCTION match_chat_memories (
    query_embedding VECTOR(1536),
    match_threshold FLOAT,
    match_count INT,
    filter_chat_id BIGINT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    chat_id BIGINT,
    sender_id BIGINT,
    sender_name TEXT,
    is_owner BOOLEAN,
    role TEXT,
    message_type TEXT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT,
    created_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        cm.id,
        cm.chat_id,
        cm.sender_id,
        cm.sender_name,
        cm.is_owner,
        cm.role,
        cm.message_type,
        cm.content,
        cm.metadata,
        1 - (cm.embedding <=> query_embedding) AS similarity,
        cm.created_at
    FROM chat_memory cm
    WHERE 
        (filter_chat_id IS NULL OR cm.chat_id = filter_chat_id)
        AND 1 - (cm.embedding <=> query_embedding) >= match_threshold
    ORDER BY cm.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 6. Configure Row Level Security (RLS)
-- If using service_role key, it bypasses RLS. For anon key, grant select/insert:
ALTER TABLE chat_memory ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow service role full access" 
ON chat_memory 
FOR ALL 
TO service_role 
USING (true) 
WITH CHECK (true);

CREATE POLICY "Allow authenticated and anon access" 
ON chat_memory 
FOR ALL 
TO anon, authenticated 
USING (true) 
WITH CHECK (true);
