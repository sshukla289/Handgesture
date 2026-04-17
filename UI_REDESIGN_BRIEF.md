# Gesture to Language UI Redesign Brief

## Product concept
The product should feel like a premium computer-vision control room rather than a generic admin dashboard. The camera is the hero surface, prediction is the primary outcome, and training behaves like a background intelligence task that upgrades the system without freezing the workspace.

## Page structure
1. Header
   Brand, product title, backend status, model readiness, theme toggle.
2. KPI strip
   Stream state, model readiness, sample volume, label count, training state.
3. Main workspace
   Large live camera panel on the left.
   Prediction card, training card, and collection card stacked on the right.
4. Insights row
   Recognition log on the left.
   Label cloud and training-source summary on the right.

## Visual direction
- Dark-mode-first AI workspace with glassy surfaces and restrained gradients.
- Large expressive headings with tighter letter spacing.
- Soft cyan-blue accent system with green success and warm warning states.
- Rounded surfaces, subtle glow, and light motion only where feedback matters.
- Dense but readable composition with very little dead space.

## Design system
### Colors
- Background: `#08111F`
- Background elevated: `#0D1727`
- Panel surface: `rgba(11, 19, 32, 0.88)`
- Strong surface: `rgba(15, 27, 45, 0.96)`
- Border: `rgba(145, 166, 196, 0.14)`
- Primary text: `#F3F7FF`
- Secondary text: `#97A7BF`
- Accent: `#5DD6FF`
- Accent strong: `#1FA7FF`
- Success: `#31D6A1`
- Warning: `#FFC86F`
- Error: `#FF7F86`

### Typography
- Primary typeface: `Sora`
- Heading style: large, condensed tracking, heavy weight, short line lengths.
- Body style: medium weight, comfortable line height, muted secondary copy.
- Label style: uppercase utility labels with wide tracking.

### Spacing
- Page padding: `24px` desktop, `14px` mobile.
- Panel padding: `22px`.
- Internal card spacing: `14px`, `16px`, `18px`, `22px`.
- Corner radius: `28px` panels, `20px` internal cards, `999px` chips.

### Cards
- Use layered dark surfaces with one gradient highlight and one soft top edge.
- Prioritize hierarchy through scale and contrast, not by adding more borders.
- Internal metric cards should feel like compact glass capsules, not flat boxes.

### Buttons
- Primary button uses a cyan-to-blue gradient with dark text for contrast.
- Secondary button stays tonal and low-contrast until hover.
- Hover motion should be limited to a subtle `translateY(-1px)`.
- Disabled controls should drop opacity but stay readable.

### Inputs
- Use filled dark inputs with soft borders and a cyan focus ring.
- Keep labels above fields to improve scannability.
- Source selection should stay near the training action.

### Status chips
- Neutral: translucent surface with muted border.
- Success: green tint and green border.
- Training/loading: accent tint and accent border.
- Warning: warm amber tint.
- Error: soft red tint.

### Progress bars
- Thin rounded track with a bright accent fill.
- Always pair progress with percent, elapsed time, and current stage text.
- Show step chips for queue, loading, training, refresh, live.

### Logs and tables
- Use a polished activity-feed/table hybrid.
- Keep three columns only: gesture, confidence, timestamp.
- Empty state should explain what action unlocks the feed.

## State design
### Idle
- Camera overlay explains the first action.
- Prediction card says "No prediction".
- Training card shows ready state and source selector.

### Loading
- Camera overlay shows startup copy.
- Prediction switches to "Scanning live frame".
- Training uses queued or loading stage with progress feedback.

### Training
- Only training actions lock.
- Camera, logs, prediction, and layout remain active.
- Stage text updates through scanning, loading, training, evaluating, saving, refreshing.

### Success
- Training chip turns green.
- Message confirms model deployment and dataset used.
- Labels and source metadata refresh automatically.

### Error
- Error chip turns red.
- Training message explains failure reason.
- Other dashboard controls remain available.

## Empty states
- No camera: "Start the camera to begin live gesture recognition."
- No prediction: "Top matches will appear here once the model sees a hand."
- No logs: "Start the camera to begin logging gesture history."
- No labels: "Train the model to populate the live label library."

## Live training UX rules
- Training must be launched as a background job.
- UI should show immediate click feedback within the same frame.
- Progress should be stage-driven, not fake animation-only feedback.
- Keep camera feed and prediction polling active during training.
- Refresh the in-memory model after training so live inference adopts the new model without a full page reset.
- Prevent dataset capture while training to avoid mutating data mid-train.

## Real-time workflow guidance
1. Start camera.
2. Confirm stream and model readiness from top-level chips.
3. Collect samples for a named gesture.
4. Stop and save collection.
5. Train model in the background.
6. Watch prediction continue while the model refreshes.
7. Review recent activity and label coverage.

## Performance guidelines
- Keep frontend dependency-free.
- Use one bootstrap payload on load.
- Poll prediction more frequently than global stats.
- Avoid overlapping requests with in-flight guards.
- Re-render logs, labels, and sources only when signatures actually change.
- Keep animations to opacity, transform, and width changes only.
- Warm-load the refreshed model after training for faster live adoption.
- Use compact DOM structures for logs and metrics.

## Responsive behavior
- Desktop-first two-column workspace.
- Collapse to one column under `1240px`.
- Reduce KPI density to two columns under `900px`.
- Stack metric grids and table columns on narrow mobile layouts.

## Recommended micro-interactions
- Single-line button lift on hover.
- Progress bar width transition.
- Toast entry fade and rise.
- Status-chip tone changes on success, training, and failure.
- No heavy parallax, particle systems, or high-frequency animation loops.
