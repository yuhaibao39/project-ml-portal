export const meta = {
  name: 'plan-portal-architecture',
  description: 'Design architecture for multi-app Next.js portal',
  phases: [{ title: 'Plan' }],
};

phase('Plan');

const plan = await agent(
  'You are a software architect. Design a comprehensive architecture plan for:\n\n' +
  '**Multi-Application Next.js Portal** with:\n' +
  '1. **Next.js Portal** (App Router, Tailwind CSS) - unified navigation between two apps\n' +
  '2. **App 1: Property Value Estimator** - Python FastAPI backend + ML model integration\n' +
  '3. **App 2: Property Market Analysis** - Java Spring Boot 3.4.4 backend + ML model integration\n\n' +
  'The ML model is a housing price regression model trained on the California housing dataset ' +
  '(features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude). ' +
  'Both backends need to call this model for predictions.\n\n' +
  'Output a structured plan covering:\n' +
  '1. Directory structure (full tree)\n' +
  '2. Component tree for Next.js (client/server component split)\n' +
  '3. Data flow diagrams (text-based)\n' +
  '4. API route design for both backends\n' +
  '5. State management approach\n' +
  '6. Key shared types/interfaces\n' +
  '7. Build/deployment strategy\n\n' +
  'Keep it practical and implementable. Focus on what makes this demo-ready and impressive for an interview.',
  {
    schema: {
      type: 'object',
      properties: {
        directoryStructure: { type: 'string' },
        componentTree: { type: 'string' },
        dataFlow: { type: 'string' },
        apiRoutes: { type: 'string' },
        sharedTypes: { type: 'string' },
        implementationOrder: { type: 'string' },
      },
      required: ['directoryStructure', 'componentTree', 'dataFlow', 'apiRoutes', 'implementationOrder'],
    },
  }
);

return plan;
