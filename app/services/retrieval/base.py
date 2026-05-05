from app.schemas.common import SourceChunk


class BaseRetriever:
    def search(self, *, query: str, module: str, limit: int = 4) -> list[SourceChunk]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError
