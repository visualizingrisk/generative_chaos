import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


plt.style.use('dark_background')
NEON_COLOR = '#00FFFF'  # Cyan
BG_COLOR = '#000000'    # Pure black for max contrast
PARTICLES = 400         
STEPS = 100             # trail length
NOISE_STRENGTH = 0.1    
ATTRACTION = 0.18       


v_points = np.array([[-1.8, 1.2], [-0.8, -1.2], [0.2, 1.2]])
r_points = np.array([[0.8, -1.2], [0.8, 1.2], [2.0, 1.2], [2.0, 0], [0.8, 0], [2.2, -1.2]])

def get_target_points(num_particles, points):
    targets = []
    total_length = 0
    segments = []

    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        length = np.linalg.norm(p2 - p1)
        segments.append((p1, p2, length))
        total_length += length

    particles_assigned = 0
    for p1, p2, length in segments:
        count = int((length / total_length) * num_particles)
        if count == 0: continue
        t = np.linspace(0, 1, count)
        segment_targets = p1[np.newaxis, :] * (1 - t)[:, np.newaxis] + p2[np.newaxis, :] * t[:, np.newaxis]
        targets.append(segment_targets)
        particles_assigned += count
    

    remaining = num_particles - particles_assigned
    if remaining > 0 and segments:
         p1, p2, _ = segments[-1]
         t = np.linspace(0, 1, remaining)
         segment_targets = p1[np.newaxis, :] * (1 - t)[:, np.newaxis] + p2[np.newaxis, :] * t[:, np.newaxis]
         targets.append(segment_targets)

    return np.vstack(targets) if targets else np.zeros((num_particles, 2))


v_targets = get_target_points(PARTICLES // 2, v_points)
r_targets = get_target_points(PARTICLES // 2, r_points)
all_targets = np.vstack([v_targets, r_targets])
actual_particles = len(all_targets)


positions = all_targets + np.random.randn(actual_particles, 2) * 0.2
history = np.zeros((STEPS, actual_particles, 2))


for i in range(STEPS):
    force = all_targets - positions
    positions += ATTRACTION * force + np.random.randn(actual_particles, 2) * NOISE_STRENGTH
    history[i] = positions


fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

lines = []
for p in range(actual_particles):
    points = history[:, p, :].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lines.extend(segments)


lc_glow = LineCollection(lines, colors=NEON_COLOR, linewidths=2.5, alpha=0.1)
ax.add_collection(lc_glow)


lc_core = LineCollection(lines, colors='#E0FFFF', linewidths=0.6, alpha=0.6) 
ax.add_collection(lc_core)

ax.set_xlim(-3.0, 3.5)
ax.set_ylim(-2.5, 2.5)
ax.axis('off')

plt.savefig('vr_sharp_neon_logo.png', dpi=300, bbox_inches='tight', facecolor=BG_COLOR, pad_inches=0.1)
print("Generated vr_sharp_neon_logo.png")
plt.show()