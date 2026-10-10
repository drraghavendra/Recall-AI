import boto3

from recall.face.recognizer import SearchResult


class RekognitionRecognizer:
    """AWS Rekognition face collection. Our face_id is stored as the Rekognition ExternalImageId."""

    def __init__(self, collection_id: str, region: str, match_threshold: float):
        self._client = boto3.client("rekognition", region_name=region)
        self._collection_id = collection_id
        self._match_threshold = match_threshold

    def ensure_collection(self) -> None:
        try:
            self._client.create_collection(CollectionId=self._collection_id)
        except self._client.exceptions.ResourceAlreadyExistsException:
            pass

    def search(self, image: bytes) -> SearchResult:
        # SearchFacesByImage only searches the largest face — that is our focused-face heuristic.
        try:
            response = self._client.search_faces_by_image(
                CollectionId=self._collection_id,
                Image={"Bytes": image},
                MaxFaces=1,
                FaceMatchThreshold=self._match_threshold,
            )
        except self._client.exceptions.InvalidParameterException:
            return SearchResult(face_found=False, face_id=None, similarity=0.0)
        matches = response.get("FaceMatches", [])
        if not matches:
            return SearchResult(face_found=True, face_id=None, similarity=0.0)
        best = matches[0]
        return SearchResult(
            face_found=True, face_id=best["Face"]["ExternalImageId"], similarity=best["Similarity"]
        )

    def enroll(self, image: bytes, face_id: str) -> str | None:
        response = self._client.index_faces(
            CollectionId=self._collection_id,
            Image={"Bytes": image},
            ExternalImageId=face_id,
            MaxFaces=1,
            QualityFilter="AUTO",
            DetectionAttributes=[],
        )
        records = response.get("FaceRecords", [])
        return records[0]["Face"]["FaceId"] if records else None
