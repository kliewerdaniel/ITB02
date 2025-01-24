// story_editor.jsx
import ReactFlow, { Controls } from 'reactflow';
import { useStore } from './store';

export default function NarrativeGraph() {
  const nodes = useStore(state => state.nodes);
  const edges = useStore(state => state.edges);

  return (
    <ReactFlow 
      nodes={nodes}
      edges={edges}
      fitView
    >
      <Controls />
    </ReactFlow>
  );
}
