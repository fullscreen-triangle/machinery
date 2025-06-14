"""
Network analysis for extraordinary information patterns.

This module provides network and graph analysis capabilities to identify
extraordinary patterns in connected data structures.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from collections import defaultdict
import networkx as nx


class NetworkAnalyzer:
    """Analyzer for network patterns and structures."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
    
    def add_node(self, node_id: str, data: Any = None, **attributes) -> None:
        """Add a node to the network."""
        self.graph.add_node(node_id, data=data, **attributes)
        self.directed_graph.add_node(node_id, data=data, **attributes)
    
    def add_edge(self, source: str, target: str, weight: float = 1.0, **attributes) -> None:
        """Add an edge to the network."""
        self.graph.add_edge(source, target, weight=weight, **attributes)
        self.directed_graph.add_edge(source, target, weight=weight, **attributes)
    
    def analyze_centrality(self) -> Dict[str, Dict[str, float]]:
        """Analyze various centrality measures."""
        try:
            centrality_measures = {}
            
            if len(self.graph.nodes()) > 0:
                # Degree centrality
                centrality_measures['degree'] = nx.degree_centrality(self.graph)
                
                # Betweenness centrality
                centrality_measures['betweenness'] = nx.betweenness_centrality(self.graph)
                
                # Closeness centrality
                centrality_measures['closeness'] = nx.closeness_centrality(self.graph)
                
                # PageRank (for directed graph)
                if len(self.directed_graph.nodes()) > 0:
                    centrality_measures['pagerank'] = nx.pagerank(self.directed_graph)
            
            return centrality_measures
            
        except Exception as e:
            self.logger.error(f"Centrality analysis failed: {str(e)}")
            return {}
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect communities in the network."""
        try:
            # Use Louvain algorithm for community detection
            import community
            partition = community.best_partition(self.graph)
            return partition
            
        except ImportError:
            self.logger.warning("Community detection library not available")
            return {}
        except Exception as e:
            self.logger.error(f"Community detection failed: {str(e)}")
            return {}
    
    def find_anomalous_nodes(self, threshold: float = 2.0) -> List[str]:
        """Find nodes with anomalous network properties."""
        try:
            anomalous_nodes = []
            
            if len(self.graph.nodes()) < 3:
                return anomalous_nodes
            
            # Calculate degree distribution
            degrees = [self.graph.degree(node) for node in self.graph.nodes()]
            mean_degree = np.mean(degrees)
            std_degree = np.std(degrees)
            
            # Find nodes with unusual degree
            for node in self.graph.nodes():
                degree = self.graph.degree(node)
                z_score = abs(degree - mean_degree) / std_degree if std_degree > 0 else 0
                
                if z_score > threshold:
                    anomalous_nodes.append(node)
            
            return anomalous_nodes
            
        except Exception as e:
            self.logger.error(f"Anomalous node detection failed: {str(e)}")
            return []


class GraphAnomalyDetector:
    """Detector for graph-based anomalies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    async def detect_structural_anomalies(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Detect structural anomalies in the graph."""
        try:
            anomalies = []
            
            if len(graph.nodes()) == 0:
                return anomalies
            
            # Detect isolated nodes
            isolated_nodes = list(nx.isolates(graph))
            for node in isolated_nodes:
                anomalies.append({
                    'type': 'isolated_node',
                    'node': node,
                    'score': 0.8,
                    'description': 'Node with no connections'
                })
            
            # Detect high-degree nodes (hubs)
            degrees = dict(graph.degree())
            mean_degree = np.mean(list(degrees.values()))
            std_degree = np.std(list(degrees.values()))
            
            for node, degree in degrees.items():
                if std_degree > 0:
                    z_score = (degree - mean_degree) / std_degree
                    if z_score > 2.0:  # High-degree anomaly
                        anomalies.append({
                            'type': 'high_degree_node',
                            'node': node,
                            'score': min(1.0, z_score / 5.0),
                            'degree': degree,
                            'description': f'High-degree node with {degree} connections'
                        })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Structural anomaly detection failed: {str(e)}")
            return []


class CommunityDetector:
    """Detector for community structures and anomalies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def detect_communities(self, graph: nx.Graph) -> Dict[str, Any]:
        """Detect communities and analyze their properties."""
        try:
            result = {
                'communities': {},
                'modularity': 0.0,
                'num_communities': 0,
                'anomalous_communities': []
            }
            
            if len(graph.nodes()) < 2:
                return result
            
            # Simple community detection using connected components
            communities = list(nx.connected_components(graph))
            
            result['num_communities'] = len(communities)
            
            # Analyze each community
            for i, community in enumerate(communities):
                community_size = len(community)
                subgraph = graph.subgraph(community)
                
                result['communities'][i] = {
                    'nodes': list(community),
                    'size': community_size,
                    'density': nx.density(subgraph) if community_size > 1 else 0.0
                }
                
                # Identify anomalous communities
                if community_size == 1:
                    result['anomalous_communities'].append({
                        'community_id': i,
                        'type': 'singleton',
                        'nodes': list(community),
                        'score': 0.7
                    })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Community detection failed: {str(e)}")
            return {'communities': {}, 'modularity': 0.0, 'num_communities': 0, 'anomalous_communities': []}


class InfluenceAnalyzer:
    """Analyzer for influence patterns in networks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def analyze_influence_patterns(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze influence patterns in directed networks."""
        try:
            influence_data = {
                'influencers': {},
                'influenced': {},
                'influence_chains': [],
                'anomalous_patterns': []
            }
            
            if len(graph.nodes()) == 0:
                return influence_data
            
            # Calculate in-degree and out-degree
            in_degrees = dict(graph.in_degree())
            out_degrees = dict(graph.out_degree())
            
            # Identify influencers (high out-degree, low in-degree)
            for node in graph.nodes():
                out_deg = out_degrees.get(node, 0)
                in_deg = in_degrees.get(node, 0)
                
                if out_deg > 0:
                    influence_ratio = out_deg / (in_deg + 1)  # Add 1 to avoid division by zero
                    
                    if influence_ratio > 2.0:  # Strong influencer
                        influence_data['influencers'][node] = {
                            'out_degree': out_deg,
                            'in_degree': in_deg,
                            'influence_ratio': influence_ratio
                        }
                
                # Identify highly influenced nodes
                if in_deg > out_deg and in_deg > 2:
                    influence_data['influenced'][node] = {
                        'in_degree': in_deg,
                        'out_degree': out_deg
                    }
            
            return influence_data
            
        except Exception as e:
            self.logger.error(f"Influence analysis failed: {str(e)}")
            return {'influencers': {}, 'influenced': {}, 'influence_chains': [], 'anomalous_patterns': []} 