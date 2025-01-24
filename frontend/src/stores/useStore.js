// frontend/src/stores/useStore.js
import create from 'zustand';

export const useStore = create((set) => ({
  nodes: [],
  edges: [],
  addNode: (node) => set((state) => ({ nodes: [...state.nodes, node] })),
  addEdge: (edge) => set((state) => ({ edges: [...state.edges, edge] })),
  setStory: (story) => {
    const nodes = story.map((chapter, index) => ({
      id: `chapter-${index}`,
      type: 'default',
      data: { label: `Chapter ${index + 1}` },
      position: { x: index * 250, y: 0 }
    }));
    
    const edges = nodes.slice(0, -1).map((node, index) => ({
      id: `edge-${index}`,
      source: node.id,
      target: nodes[index + 1].id
    }));

    set({ nodes, edges });
  }
}));