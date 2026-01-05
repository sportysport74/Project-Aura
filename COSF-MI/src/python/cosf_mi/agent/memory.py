"""
COSF Memory System: Long-term memory with φ-resonant retrieval

Uses vector database with harmonic similarity matching.
"""

import time
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: chromadb not installed. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

from ..core.constants import PHI


class COSFMemory:
    """Long-term memory with φ-resonant retrieval.

    Features:
    - Vector database storage (Chroma)
    - φ-harmonic similarity matching
    - Obsidian markdown sync
    - Temporal decay
    """

    def __init__(
        self,
        db_path: str = "memory/chroma",
        encoder_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize memory system.

        Args:
            db_path: Path to Chroma database
            encoder_model: SentenceTransformer model name
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb required. Install: pip install chromadb")
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="cosf_memory",
            metadata={"description": "COSF Jarvis long-term memory"}
        )

        # Initialize encoder
        self.encoder = SentenceTransformer(encoder_model)

        self.phi = PHI

    def store(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "user"
    ):
        """Store memory with φ-decomposition.

        Args:
            text: Text to store
            metadata: Optional metadata dict
            source: Source of memory (user, system, obsidian, etc.)
        """
        # Generate embedding
        embedding = self.encoder.encode(text)

        # φ-decompose for harmonic storage
        phi_components = self._phi_decompose(embedding)

        # Create metadata
        # Chromadb only accepts str, int, float, bool in metadata
        memory_metadata = {
            'timestamp': time.time(),
            'source': source,
            'phi_r0': phi_components[0],
            'phi_r1': phi_components[1],
            'phi_r2': phi_components[2],
            'phi_r3': phi_components[3],
            'phi_r4': phi_components[4],
            'length': len(text),
        }

        if metadata:
            # Sanitize metadata: chromadb only accepts str, int, float, bool
            for key, value in metadata.items():
                if isinstance(value, list):
                    # Convert lists to comma-separated strings
                    memory_metadata[key] = ','.join(str(v) for v in value)
                elif isinstance(value, (str, int, float, bool)):
                    memory_metadata[key] = value
                # Skip other types

        # Generate unique ID
        memory_id = hashlib.sha256(
            f"{text}{time.time()}".encode()
        ).hexdigest()[:16]

        # Store in Chroma
        self.collection.add(
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[memory_metadata],
            ids=[memory_id]
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        phi_weighted: bool = True,
        time_decay: bool = True,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve memories with φ-resonant scoring.

        Args:
            query: Query text
            top_k: Number of results
            phi_weighted: Use φ-harmonic re-ranking
            time_decay: Apply temporal decay
            source_filter: Optional source filter

        Returns:
            List of memory dictionaries
        """
        # Encode query
        query_embedding = self.encoder.encode(query)

        # Initial retrieval (over-fetch for re-ranking)
        where_filter = None
        if source_filter:
            where_filter = {"source": source_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k * 3, 100),  # Over-fetch
            where=where_filter
        )

        # Re-rank with φ-resonance
        if phi_weighted and results['documents'][0]:
            results = self._phi_rerank(results, query_embedding, time_decay)

        # Format results
        memories = []
        for i in range(min(top_k, len(results['documents'][0]))):
            memory = {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'id': results['ids'][0][i]
            }
            memories.append(memory)

        return memories

    def _phi_decompose(self, vector: np.ndarray) -> List[float]:
        """Decompose vector into φ-harmonic components.

        Args:
            vector: Embedding vector

        Returns:
            List of φ-harmonic projections
        """
        components = []

        # Project onto first 5 φ-harmonic basis vectors
        for i in range(5):
            # Create φ-scaled basis vector with safe scaling
            # Use normalized index to prevent overflow
            basis = np.array([
                self.phi ** (i * j / len(vector)) for j in range(len(vector))
            ])

            # Normalize
            basis = basis / np.linalg.norm(basis)

            # Project
            projection = np.dot(vector, basis)
            components.append(float(projection))

        return components

    def _phi_rerank(
        self,
        results: Dict,
        query_embedding: np.ndarray,
        time_decay: bool
    ) -> Dict:
        """Re-rank results by φ-resonance.

        Args:
            results: Initial Chroma results
            query_embedding: Query embedding vector
            time_decay: Apply temporal decay

        Returns:
            Re-ranked results
        """
        # Get query φ-components
        query_phi = self._phi_decompose(query_embedding)

        # Compute resonance scores
        resonance_scores = []
        current_time = time.time()

        for i, metadata in enumerate(results['metadatas'][0]):
            # Reconstruct phi components from individual fields
            stored_phi = [
                metadata.get('phi_r0', 0.0),
                metadata.get('phi_r1', 0.0),
                metadata.get('phi_r2', 0.0),
                metadata.get('phi_r3', 0.0),
                metadata.get('phi_r4', 0.0),
            ]

            # Harmonic similarity (weighted by powers of φ)
            resonance = sum(
                q * s * (self.phi ** j)
                for j, (q, s) in enumerate(zip(query_phi, stored_phi))
            )

            # Apply time decay if requested
            if time_decay and 'timestamp' in metadata:
                age_days = (current_time - metadata['timestamp']) / 86400
                # Decay factor: older memories have less weight
                decay = np.exp(-age_days / 30)  # 30-day half-life
                resonance *= decay

            resonance_scores.append(resonance)

        # Combine with original distance (lower distance = better)
        # Higher resonance = better
        combined_scores = []
        for i, (dist, res) in enumerate(zip(results['distances'][0], resonance_scores)):
            # Normalize distance to [0, 1]
            norm_dist = 1 / (1 + dist)

            # Combine (equal weight for now)
            combined = norm_dist * 0.5 + res * 0.5
            combined_scores.append(combined)

        # Sort by combined score (descending)
        sorted_indices = np.argsort(combined_scores)[::-1]

        # Reorder results
        for key in results:
            if results[key] and results[key][0]:  # Check if not empty
                results[key][0] = [results[key][0][i] for i in sorted_indices]

        return results

    def sync_with_obsidian(
        self,
        vault_path: str,
        file_pattern: str = "*.md"
    ):
        """Sync memory with Obsidian markdown vault.

        Args:
            vault_path: Path to Obsidian vault
            file_pattern: Glob pattern for files (default: *.md)
        """
        vault_path = Path(vault_path)

        if not vault_path.exists():
            print(f"Warning: Vault path {vault_path} does not exist")
            return

        # Find all markdown files
        md_files = list(vault_path.rglob(file_pattern))

        print(f"Found {len(md_files)} files in {vault_path}")

        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract Obsidian tags
                tags = self._extract_obsidian_tags(content)

                # Store with metadata
                self.store(
                    content,
                    metadata={
                        'file': str(md_file),
                        'tags': tags,
                        'filename': md_file.name,
                    },
                    source='obsidian'
                )

                print(f"  Synced: {md_file.name}")

            except Exception as e:
                print(f"  Error syncing {md_file.name}: {e}")

        print(f"\nSynced {len(md_files)} Obsidian files to memory")

    def _extract_obsidian_tags(self, content: str) -> List[str]:
        """Extract tags from Obsidian markdown.

        Args:
            content: Markdown content

        Returns:
            List of tags
        """
        import re

        # Find #tags
        tags = re.findall(r'#(\w+)', content)

        # Find YAML frontmatter tags
        frontmatter_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            tag_match = re.search(r'tags:\s*\[(.*?)\]', frontmatter)
            if tag_match:
                yaml_tags = [t.strip().strip('"\'') for t in tag_match.group(1).split(',')]
                tags.extend(yaml_tags)

        return list(set(tags))  # Remove duplicates

    def search_by_tag(self, tag: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search memories by Obsidian tag.

        Args:
            tag: Tag to search for
            top_k: Number of results

        Returns:
            List of matching memories
        """
        # Query with metadata filter
        results = self.collection.get(
            where={"tags": {"$contains": tag}},
            limit=top_k
        )

        # Format results
        memories = []
        for i in range(len(results['documents'])):
            memory = {
                'text': results['documents'][i],
                'metadata': results['metadatas'][i],
                'id': results['ids'][i]
            }
            memories.append(memory)

        return memories

    def clear(self):
        """Clear all memories (use with caution!)."""
        self.client.delete_collection("cosf_memory")
        self.collection = self.client.get_or_create_collection(
            name="cosf_memory",
            metadata={"description": "COSF Jarvis long-term memory"}
        )

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary of statistics
        """
        count = self.collection.count()

        return {
            'total_memories': count,
            'db_path': str(self.db_path),
            'phi': self.phi,
        }
