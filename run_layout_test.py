import asyncio
from shapely.geometry import box

from backend.agents.layout_agent import LayoutGenerationAgent
from backend.core.fbsl_models import FBSLLayoutNode, Function


async def main():
    agent = LayoutGenerationAgent()
    node = FBSLLayoutNode()

    # Create 20 individual workstation functions (small enclosed offices)
    for i in range(20):
        f = Function(
            name=f"provide_workstation_{i+1}",
        )
        f.spatial_requirements = {'preferred_area': 6.0}
        node.functions[f.function_id] = f

    # Meeting rooms: 4, 6, 8 person capacities
    m1 = Function(name='provide_meeting_room_small')
    m1.spatial_requirements = {'preferred_area': 8.0}
    node.functions[m1.function_id] = m1

    m2 = Function(name='provide_meeting_room_medium')
    m2.spatial_requirements = {'preferred_area': 12.0}
    node.functions[m2.function_id] = m2

    m3 = Function(name='provide_meeting_room_large')
    m3.spatial_requirements = {'preferred_area': 16.0}
    node.functions[m3.function_id] = m3

    # Collaboration area, break room, storage
    collab = Function(name='provide_collaboration_area')
    collab.spatial_requirements = {'preferred_area': 25.0}
    node.functions[collab.function_id] = collab

    break_room = Function(name='provide_break_room')
    break_room.spatial_requirements = {'preferred_area': 15.0}
    node.functions[break_room.function_id] = break_room

    storage = Function(name='provide_storage')
    storage.spatial_requirements = {'preferred_area': 10.0}
    node.functions[storage.function_id] = storage

    # Optional metadata
    node.metadata['project_name'] = 'Test Office 30p - 25x16'

    site = box(0, 0, 25, 16)  # 25m x 16m site = 400 m²

    layout = await agent.generate_layout(node, site_boundary=site)

    # Print concise metrics
    print('Layout metrics:')
    print(f" total_area = {layout.total_area}")
    print(f" used_area  = {layout.used_area}")
    print(f" circulation_area = {layout.circulation_area}")
    print(f" space_utilization_ratio = {layout.space_utilization_ratio}")
    print(f" circulation_efficiency = {layout.circulation_efficiency}")
    print(f" adjacency_satisfaction_score = {layout.adjacency_satisfaction_score}")
    print(f" compactness_score = {layout.compactness_score}")

    # Also print number of rooms and whether adjacency matrix present
    print(f" rooms (before to_dict) = {len(layout.rooms)}")
    print(f" room_polygons = {len(layout.room_polygons) if hasattr(layout, 'room_polygons') else 'N/A'}")
    print(f" adjacency_matrix = {layout.adjacency_matrix.shape if layout.adjacency_matrix is not None else 'None'}")
    
    # Now serialize
    d = layout.to_dict()
    print(f" rooms (in dict) = {len(d.get('rooms', {}))}")


if __name__ == '__main__':
    asyncio.run(main())
