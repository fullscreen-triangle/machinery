"""
Puzzle generation and management for context validation.

This module contains various types of puzzles that the system can generate
and present to validate context understanding and coherence.
"""

import logging
import random
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio


@dataclass
class PuzzleQuestion:
    """A single question within a puzzle."""
    id: str
    question_type: str
    question_text: str
    expected_answer: Any
    context_clues: List[str] = field(default_factory=list)
    difficulty: str = "normal"
    weight: float = 1.0


@dataclass
class PuzzleSolution:
    """Solution to a puzzle."""
    puzzle_id: str
    answers: Dict[str, Any]
    is_correct: bool
    confidence: float
    context_evidence: Dict[str, Any] = field(default_factory=dict)
    solving_method: str = "unknown"


class BasePuzzle(ABC):
    """Base class for all puzzle types."""
    
    def __init__(self, puzzle_id: str, puzzle_type: str, difficulty: str = "normal"):
        self.id = puzzle_id
        self.type = puzzle_type
        self.difficulty = difficulty
        self.created_at = datetime.now()
        self.questions: List[PuzzleQuestion] = []
        self.metadata: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def generate_questions(self, context: Dict[str, Any], **kwargs) -> List[PuzzleQuestion]:
        """Generate questions for this puzzle type."""
        pass
    
    def add_question(self, question: PuzzleQuestion) -> None:
        """Add a question to the puzzle."""
        self.questions.append(question)
    
    def validate_solution(self, solution: PuzzleSolution) -> bool:
        """Validate if the provided solution is correct."""
        if solution.puzzle_id != self.id:
            return False
        
        correct_answers = 0
        total_weight = 0
        
        for question in self.questions:
            expected = question.expected_answer
            provided = solution.answers.get(question.id)
            
            if self._compare_answers(expected, provided):
                correct_answers += question.weight
            
            total_weight += question.weight
        
        if total_weight == 0:
            return False
        
        accuracy = correct_answers / total_weight
        return accuracy >= 0.7  # 70% threshold for correctness
    
    def _compare_answers(self, expected: Any, provided: Any) -> bool:
        """Compare expected and provided answers."""
        if expected == provided:
            return True
        
        # Handle string comparisons (case-insensitive)
        if isinstance(expected, str) and isinstance(provided, str):
            return expected.lower().strip() == provided.lower().strip()
        
        # Handle numeric comparisons with tolerance
        if isinstance(expected, (int, float)) and isinstance(provided, (int, float)):
            return abs(expected - provided) < 0.01
        
        return False


class ContextPuzzle(BasePuzzle):
    """Puzzle that tests understanding of recent context."""
    
    def __init__(self, puzzle_id: str, difficulty: str = "normal"):
        super().__init__(puzzle_id, "context_puzzle", difficulty)
    
    def generate_questions(self, context: Dict[str, Any], **kwargs) -> List[PuzzleQuestion]:
        """Generate context-based questions."""
        questions = []
        processes = kwargs.get('processes', [])
        
        if not processes:
            return questions
        
        # Question 1: Recall recent process ID
        if processes:
            recent_process = processes[-1]
            questions.append(PuzzleQuestion(
                id=f"{self.id}_q1",
                question_type="recall",
                question_text="What was the ID of the most recent process?",
                expected_answer=recent_process.process_id,
                context_clues=["recent_process", "process_id"],
                difficulty=self.difficulty,
                weight=1.0
            ))
        
        # Question 2: Count processes
        questions.append(PuzzleQuestion(
            id=f"{self.id}_q2",
            question_type="count",
            question_text="How many processes have been recorded?",
            expected_answer=len(processes),
            context_clues=["process_count", "total_processes"],
            difficulty=self.difficulty,
            weight=1.0
        ))
        
        # Question 3: Coherence analysis
        if processes:
            avg_coherence = sum(p.coherence_score for p in processes) / len(processes)
            questions.append(PuzzleQuestion(
                id=f"{self.id}_q3",
                question_type="analysis",
                question_text="What is the average coherence score (rounded to 2 decimals)?",
                expected_answer=round(avg_coherence, 2),
                context_clues=["coherence_score", "average"],
                difficulty=self.difficulty,
                weight=1.5
            ))
        
        self.questions = questions
        return questions


class LogicPuzzle(BasePuzzle):
    """Puzzle that tests logical reasoning capabilities."""
    
    def __init__(self, puzzle_id: str, difficulty: str = "normal"):
        super().__init__(puzzle_id, "logic_puzzle", difficulty)
    
    def generate_questions(self, context: Dict[str, Any], **kwargs) -> List[PuzzleQuestion]:
        """Generate logic-based questions."""
        questions = []
        
        # Simple pattern recognition puzzle
        sequence = self._generate_sequence()
        questions.append(PuzzleQuestion(
            id=f"{self.id}_q1",
            question_type="pattern",
            question_text=f"What is the next number in this sequence: {', '.join(map(str, sequence))}?",
            expected_answer=self._calculate_next_in_sequence(sequence),
            context_clues=["pattern", "sequence", "mathematics"],
            difficulty=self.difficulty,
            weight=1.0
        ))
        
        # Logic deduction puzzle
        logic_answer = self._generate_logic_puzzle()
        questions.append(PuzzleQuestion(
            id=f"{self.id}_q2",
            question_type="deduction",
            question_text="If A > B and B > C, what is the relationship between A and C?",
            expected_answer="A > C",
            context_clues=["logic", "transitivity", "comparison"],
            difficulty=self.difficulty,
            weight=1.5
        ))
        
        self.questions = questions
        return questions
    
    def _generate_sequence(self) -> List[int]:
        """Generate a simple arithmetic sequence."""
        start = random.randint(1, 10)
        step = random.randint(1, 5)
        return [start + i * step for i in range(4)]
    
    def _calculate_next_in_sequence(self, sequence: List[int]) -> int:
        """Calculate the next number in an arithmetic sequence."""
        if len(sequence) < 2:
            return sequence[0] + 1
        
        step = sequence[1] - sequence[0]
        return sequence[-1] + step
    
    def _generate_logic_puzzle(self) -> str:
        """Generate a simple logic puzzle."""
        return "A > C"  # For the transitivity example


class MemoryPuzzle(BasePuzzle):
    """Puzzle that tests memory and recall of recent information."""
    
    def __init__(self, puzzle_id: str, difficulty: str = "normal"):
        super().__init__(puzzle_id, "memory_puzzle", difficulty)
    
    def generate_questions(self, context: Dict[str, Any], **kwargs) -> List[PuzzleQuestion]:
        """Generate memory-based questions."""
        questions = []
        processes = kwargs.get('processes', [])
        
        if not processes:
            return questions
        
        # Memory test: recall specific process details
        if len(processes) >= 2:
            target_process = processes[-2]  # Second to last process
            questions.append(PuzzleQuestion(
                id=f"{self.id}_q1",
                question_type="recall",
                question_text="What was the coherence score of the second-to-last process (rounded to 2 decimals)?",
                expected_answer=round(target_process.coherence_score, 2),
                context_clues=["memory", "coherence_score", "second_last"],
                difficulty=self.difficulty,
                weight=1.0
            ))
        
        # Sequence memory test
        if len(processes) >= 3:
            recent_ids = [p.process_id for p in processes[-3:]]
            questions.append(PuzzleQuestion(
                id=f"{self.id}_q2",
                question_type="sequence",
                question_text="List the IDs of the last 3 processes in order (comma-separated)",
                expected_answer=", ".join(recent_ids),
                context_clues=["sequence", "process_ids", "order"],
                difficulty=self.difficulty,
                weight=1.5
            ))
        
        self.questions = questions
        return questions


class SummaryPuzzle(BasePuzzle):
    """Puzzle that tests ability to summarize and compress information."""
    
    def __init__(self, puzzle_id: str, difficulty: str = "normal"):
        super().__init__(puzzle_id, "summary_puzzle", difficulty)
    
    def generate_questions(self, context: Dict[str, Any], **kwargs) -> List[PuzzleQuestion]:
        """Generate summary-based questions."""
        questions = []
        processes = kwargs.get('processes', [])
        
        if not processes:
            return questions
        
        # Summarization test
        total_processes = len(processes)
        avg_coherence = sum(p.coherence_score for p in processes) / len(processes) if processes else 0
        
        expected_summary = f"Processed {total_processes} items with average coherence {avg_coherence:.2f}"
        
        questions.append(PuzzleQuestion(
            id=f"{self.id}_q1",
            question_type="summarization",
            question_text="Provide a brief summary of recent processing activity (format: 'Processed X items with average coherence Y')",
            expected_answer=expected_summary,
            context_clues=["summary", "processing", "coherence"],
            difficulty=self.difficulty,
            weight=2.0
        ))
        
        self.questions = questions
        return questions


class IntegrationPuzzle(BasePuzzle):
    """Puzzle that tests integration with other system components."""
    
    def __init__(self, puzzle_id: str, difficulty: str = "normal"):
        super().__init__(puzzle_id, "integration_puzzle", difficulty)
    
    def generate_questions(self, context: Dict[str, Any], **kwargs) -> List[PuzzleQuestion]:
        """Generate integration-based questions."""
        questions = []
        
        # Test knowledge of system components
        questions.append(PuzzleQuestion(
            id=f"{self.id}_q1",
            question_type="system_knowledge",
            question_text="Which system component handles extraordinary information detection?",
            expected_answer="spectacular",
            context_clues=["system", "components", "extraordinary"],
            difficulty=self.difficulty,
            weight=1.0
        ))
        
        # Test integration awareness
        questions.append(PuzzleQuestion(
            id=f"{self.id}_q2",
            question_type="integration",
            question_text="What is the primary purpose of the nicotine system?",
            expected_answer="context validation",
            context_clues=["nicotine", "purpose", "context"],
            difficulty=self.difficulty,
            weight=1.5
        ))
        
        self.questions = questions
        return questions


class PuzzleGenerator:
    """Generator for creating various types of puzzles."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.puzzle_types = {
            'context': ContextPuzzle,
            'logic': LogicPuzzle,
            'memory': MemoryPuzzle,
            'summary': SummaryPuzzle,
            'integration': IntegrationPuzzle
        }
    
    async def generate_context_puzzle(
        self, 
        processes: List[Any], 
        context: Dict[str, Any],
        difficulty_level: str = "normal"
    ) -> Optional[ContextPuzzle]:
        """Generate a context validation puzzle."""
        try:
            puzzle_id = self._generate_puzzle_id("context")
            puzzle = ContextPuzzle(puzzle_id, difficulty_level)
            
            questions = puzzle.generate_questions(
                context=context,
                processes=processes
            )
            
            if questions:
                self.logger.info(f"Generated context puzzle with {len(questions)} questions")
                return puzzle
            else:
                self.logger.warning("Failed to generate context puzzle questions")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating context puzzle: {str(e)}")
            return None
    
    async def generate_puzzle(
        self, 
        puzzle_type: str, 
        difficulty: str = "normal",
        **kwargs
    ) -> Optional[BasePuzzle]:
        """Generate a puzzle of the specified type."""
        try:
            if puzzle_type not in self.puzzle_types:
                self.logger.error(f"Unknown puzzle type: {puzzle_type}")
                return None
            
            puzzle_id = self._generate_puzzle_id(puzzle_type)
            puzzle_class = self.puzzle_types[puzzle_type]
            puzzle = puzzle_class(puzzle_id, difficulty)
            
            context = kwargs.get('context', {})
            questions = puzzle.generate_questions(context, **kwargs)
            
            if questions:
                self.logger.info(f"Generated {puzzle_type} puzzle with {len(questions)} questions")
                return puzzle
            else:
                self.logger.warning(f"Failed to generate {puzzle_type} puzzle questions")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating {puzzle_type} puzzle: {str(e)}")
            return None
    
    def _generate_puzzle_id(self, puzzle_type: str) -> str:
        """Generate a unique puzzle ID."""
        timestamp = datetime.now().isoformat()
        content = f"{puzzle_type}_{timestamp}_{random.randint(1000, 9999)}"
        puzzle_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"puzzle_{puzzle_type}_{puzzle_hash}"
    
    def get_available_puzzle_types(self) -> List[str]:
        """Get list of available puzzle types."""
        return list(self.puzzle_types.keys())
    
    def validate_puzzle_config(self, puzzle_type: str, config: Dict[str, Any]) -> bool:
        """Validate puzzle configuration."""
        if puzzle_type not in self.puzzle_types:
            return False
        
        required_fields = ['enabled', 'difficulty_level']
        return all(field in config for field in required_fields) 