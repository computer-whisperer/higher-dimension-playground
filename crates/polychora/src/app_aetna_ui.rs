use aetna_core::prelude::{
    badge, button, card, card_content, card_header, card_title, column, image as aetna_image, mono,
    render_bundle, row, spacer, stack, text, tokens, write_bundle, Align, Axis, Color, Cursor, El,
    Image, ImageFit, Justify, Kind, Rect, Size, StyleProfile, SurfaceRole,
};
use higher_dimension_playground::vulkan_setup::vulkan_setup;
use polychora::content_registry::ContentRegistry;
use polychora::shared::inventory::Inventory;
use polychora::shared::protocol::ItemStack;
use std::{panic::Location, path::Path};

use super::{block_data_from_slot, format_cbor_for_display, App, WailaTarget};
use crate::material_icons::{self, MaterialIconSheet};

const HOTBAR_SLOT_KEY_PREFIX: &str = "aetna_hotbar_slot_";
const ORIENTATION_KEY_PREFIX: &str = "aetna_orientation_";
const NAV_HUD_BOTTOM_LEFT_RESERVED_WIDTH: f32 = 150.0;

struct HotbarSlotView {
    name: String,
    count: u32,
    color: Color,
    icon: Option<Image>,
    selected: bool,
}

pub(super) fn dump_headless_aetna_overlay_bundle(
    width: u32,
    height: u32,
    out_dir: &Path,
) -> std::io::Result<()> {
    let (content_registry, pending_texture_uploads) =
        polychora::plugin_loader::create_full_registry();
    let material_resolver =
        polychora::content_registry::MaterialResolver::from_registry(&content_registry);
    let (instance, device, queue) = vulkan_setup(None);
    let material_icon_sheet = material_icons::generate_material_icon_sheet_gpu(
        device,
        queue,
        instance,
        &content_registry,
        &material_resolver,
        &pending_texture_uploads,
    );
    let inventory = Inventory::default_creative();
    let slots = (0..9).map(|index| {
        build_aetna_hotbar_slot_for_stack(
            &content_registry,
            material_icon_sheet.as_ref(),
            index,
            inventory.hotbar_slot(index),
            index == 0,
        )
    });
    let hotbar = build_aetna_hotbar_from_slots(slots);
    let mut overlay = build_aetna_overlay_shell(
        hotbar,
        build_aetna_orientation_controls_view("Ori: 0".to_string(), false),
        None,
    );
    write_aetna_bundle(&mut overlay, width, height, out_dir, "aetna_hud")
}

fn write_aetna_bundle(
    overlay: &mut El,
    width: u32,
    height: u32,
    out_dir: &Path,
    name: &str,
) -> std::io::Result<()> {
    let viewport = Rect::new(0.0, 0.0, width as f32, height as f32);
    let bundle = render_bundle(overlay, viewport);
    let written = write_bundle(&bundle, out_dir, name)?;
    eprintln!("Wrote Aetna HUD bundle artifacts to {}", out_dir.display());
    for path in written {
        eprintln!("  {}", path.display());
    }
    if !bundle.lint.findings.is_empty() {
        eprintln!("\nAetna lint findings ({}):", bundle.lint.findings.len());
        eprint!("{}", bundle.lint.text());
    }
    Ok(())
}

impl App {
    pub(super) fn dump_aetna_overlay_bundle(&self) -> std::io::Result<()> {
        let Some(mut overlay) = self.build_aetna_overlay() else {
            return Ok(());
        };
        write_aetna_bundle(
            &mut overlay,
            self.args.width,
            self.args.height,
            &self.args.aetna_bundle_dir,
            "aetna_hud",
        )
    }

    pub(super) fn build_aetna_overlay(&self) -> Option<El> {
        let hotbar = self.build_aetna_hotbar();
        let orientation = self.build_aetna_orientation_controls();
        Some(build_aetna_overlay_shell(
            hotbar,
            orientation,
            self.build_aetna_waila_panel(),
        ))
    }

    fn build_aetna_hotbar(&self) -> El {
        let slots = (0..9).map(|i| self.build_aetna_hotbar_slot(i));
        build_aetna_hotbar_from_slots(slots)
    }

    fn build_aetna_hotbar_slot(&self, index: usize) -> El {
        build_aetna_hotbar_slot_for_stack(
            &self.content_registry,
            self.material_icon_sheet.as_ref(),
            index,
            self.inventory.hotbar_slot(index),
            index == self.hotbar_selected_index,
        )
    }

    fn build_aetna_orientation_controls(&self) -> El {
        use polychora::shared::voxel::TesseractOrientation;

        let is_rotated = self.placement_orientation != TesseractOrientation::IDENTITY;
        let label = if is_rotated {
            format!("Ori: {}", self.placement_orientation.0)
        } else {
            "Ori: 0".to_string()
        };

        build_aetna_orientation_controls_view(label, is_rotated)
    }

    fn build_aetna_waila_panel(&self) -> Option<El> {
        let target = self.waila_target.as_ref()?;
        let panel = match target {
            WailaTarget::Block { coords, block } => {
                let entry = self
                    .content_registry
                    .block_entry(block.namespace, block.block_type);
                let name = entry.map(|e| e.name.as_str()).unwrap_or("Unknown");
                let category = entry.map(|e| e.category.label()).unwrap_or("Unknown");
                let ns_label = self.content_registry.namespace_label(block.namespace);
                let scale_label = if block.scale_exp != 0 {
                    format!("  scale: {}", block.scale_exp)
                } else {
                    String::new()
                };

                waila_panel(
                    name,
                    category,
                    [
                        format!(
                            "ns: {:#010x} ({})  type: {:#010x}",
                            block.namespace, ns_label, block.block_type
                        ),
                        format!(
                            "[{}, {}, {}, {}]{}",
                            coords[0], coords[1], coords[2], coords[3], scale_label
                        ),
                    ],
                )
            }
            WailaTarget::Entity {
                entity_id,
                entity_type_ns,
                entity_type,
                position,
                orientation,
                scale,
                data,
                distance,
            } => {
                let entry = self
                    .content_registry
                    .entity_lookup(*entity_type_ns, *entity_type);
                let canonical_name = entry
                    .map(|e| e.canonical_name.as_str())
                    .unwrap_or("unknown");
                let category = entry
                    .map(|e| format!("{:?}", e.category))
                    .unwrap_or_else(|| "Unknown".to_string());
                let player_name = self.remote_players.get(entity_id).map(|p| p.name.clone());
                let display_name = if let Some(name) = player_name {
                    format!("{} ({})", canonical_name, name)
                } else {
                    canonical_name.to_string()
                };
                let ns_label = self.content_registry.namespace_label(*entity_type_ns);
                let mut lines = vec![
                    format!(
                        "id: {}  ns: {:#010x} ({})  type: {:#010x}",
                        entity_id, entity_type_ns, ns_label, entity_type
                    ),
                    format!(
                        "pos: [{:.1}, {:.1}, {:.1}, {:.1}]",
                        position[0], position[1], position[2], position[3]
                    ),
                    format!(
                        "ori: [{:.2}, {:.2}, {:.2}, {:.2}]  scale: {:.2}",
                        orientation[0], orientation[1], orientation[2], orientation[3], scale
                    ),
                ];
                if let Some(entry) = entry {
                    if let Some(config) = &entry.sim_config {
                        lines.push(format!(
                            "{:?}: {:?} spd={:.1}",
                            config.mode, config.locomotion, config.move_speed
                        ));
                    }
                }
                if let Some(decoded) = format_cbor_for_display(data) {
                    lines.push(format!("data: {}", decoded));
                }

                waila_panel(
                    display_name,
                    format!("{} {:.1}m", category, distance),
                    lines,
                )
            }
        };

        Some(panel)
    }
}

fn build_aetna_overlay_shell(hotbar: El, orientation: El, waila: Option<El>) -> El {
    let mut children = vec![hotbar, orientation];
    if let Some(waila) = waila {
        children.push(waila);
    }

    stack(children).fill_size().layout(|cx| {
        let (hotbar_w, hotbar_h) = (cx.measure)(&cx.children[0]);
        let centered_hotbar_x = cx.container.x + (cx.container.w - hotbar_w) * 0.5;
        let max_hotbar_x = (cx.container.right() - hotbar_w - 12.0).max(cx.container.x + 12.0);
        let min_hotbar_x = (cx.container.x + NAV_HUD_BOTTOM_LEFT_RESERVED_WIDTH).min(max_hotbar_x);
        let hotbar_x = centered_hotbar_x.clamp(min_hotbar_x, max_hotbar_x);
        let hotbar_y = cx.container.bottom() - hotbar_h - 12.0;
        let hotbar_rect = Rect::new(hotbar_x, hotbar_y, hotbar_w, hotbar_h);

        let mut rects = Vec::with_capacity(cx.children.len());
        for (index, child) in cx.children.iter().enumerate() {
            let (measured_w, measured_h) = (cx.measure)(child);
            let rect = match index {
                0 => hotbar_rect,
                1 => {
                    let left_x = hotbar_rect.x - measured_w - 10.0;
                    if left_x >= cx.container.x + 12.0 {
                        Rect::new(
                            left_x,
                            hotbar_rect.y + (hotbar_rect.h - measured_h) * 0.5,
                            measured_w,
                            measured_h,
                        )
                    } else {
                        Rect::new(
                            cx.container.x + ((cx.container.w - measured_w) * 0.5).max(12.0),
                            (hotbar_rect.y - measured_h - 10.0).max(cx.container.y + 12.0),
                            measured_w,
                            measured_h,
                        )
                    }
                }
                _ => {
                    let width = measured_w.min((cx.container.w - 24.0).max(260.0));
                    Rect::new(
                        cx.container.x + ((cx.container.w - width) * 0.5).max(12.0),
                        cx.container.y + 30.0,
                        width,
                        measured_h,
                    )
                }
            };
            rects.push(rect);
        }
        rects
    })
}

fn build_aetna_hotbar_from_slots(slots: impl IntoIterator<Item = El>) -> El {
    row(slots)
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .height(Size::Hug)
        .width(Size::Hug)
}

fn build_aetna_hotbar_slot_for_stack(
    content_registry: &ContentRegistry,
    material_icon_sheet: Option<&MaterialIconSheet>,
    index: usize,
    stack: &Option<ItemStack>,
    selected: bool,
) -> El {
    let (name, count, color, icon) = stack
        .as_ref()
        .map(|stack| {
            let tex = content_registry.resolve_item_thumbnail_texture(&stack.item);
            let icon =
                tex.and_then(|tex| material_icon_sheet?.aetna_image(tex.namespace, tex.texture_id));

            let (name, color) = if let Some(block) = stack.to_block_data() {
                let entry = content_registry.block_entry(block.namespace, block.block_type);
                let name = entry
                    .map(|entry| entry.name.clone())
                    .unwrap_or_else(|| "Unknown".to_string());
                let [r, g, b] = entry.map(|entry| entry.color).unwrap_or([128, 128, 128]);
                (name, Color::rgb(r, g, b))
            } else if let Some((entity_ns, entity_type)) = stack.spawn_egg_entity_key() {
                let entry = content_registry.entity_lookup(entity_ns, entity_type);
                let name = entry
                    .map(|entry| entry.canonical_name.clone())
                    .unwrap_or_else(|| "Unknown".to_string());
                let [r, g, b] = entry
                    .map(|entry| entry.base_color)
                    .unwrap_or([128, 128, 128]);
                (name, Color::rgb(r, g, b))
            } else {
                let [r, g, b] =
                    content_registry.item_color(stack.item.namespace, stack.item.item_type);
                (
                    content_registry
                        .item_name(stack.item.namespace, stack.item.item_type)
                        .to_string(),
                    Color::rgb(r, g, b),
                )
            };

            (name, stack.count, color, icon)
        })
        .unwrap_or_else(|| ("Empty".to_string(), 0, Color::rgb(44, 48, 58), None));

    build_aetna_hotbar_slot_view(
        index,
        HotbarSlotView {
            name,
            count,
            color,
            icon,
            selected,
        },
    )
}

#[track_caller]
fn build_aetna_hotbar_slot_view(index: usize, slot: HotbarSlotView) -> El {
    let label = short_label(slot.name);
    let icon = if let Some(icon) = slot.icon {
        aetna_image(icon)
            .image_fit(ImageFit::Contain)
            .width(Size::Fixed(44.0))
            .height(Size::Fixed(30.0))
            .radius(4.0)
    } else {
        column(std::iter::empty::<El>())
            .width(Size::Fixed(44.0))
            .height(Size::Fixed(30.0))
            .fill(slot.color)
            .radius(4.0)
    };

    El::new(Kind::Custom("polychora_hotbar_slot"))
        .at_loc(Location::caller())
        .style_profile(StyleProfile::Surface)
        .surface_role(SurfaceRole::Panel)
        .axis(Axis::Column)
        .children([
            row([
                text(format!("{}", index + 1)).caption().muted(),
                spacer(),
                if slot.count > 1 {
                    badge(format!("{}", slot.count)).muted()
                } else {
                    text("")
                },
            ])
            .width(Size::Fill(1.0))
            .align(Align::Center),
            icon,
            text(label)
                .caption()
                .center_text()
                .ellipsis()
                .max_lines(1)
                .width(Size::Fill(1.0)),
        ])
        .key(format!("aetna_hotbar_slot_{index}"))
        .focusable()
        .cursor(Cursor::Pointer)
        .width(Size::Fixed(76.0))
        .height(Size::Fixed(82.0))
        .padding(6.0)
        .gap(2.0)
        .align(Align::Center)
        .fill(tokens::CARD.with_alpha(205))
        .stroke(if slot.selected {
            tokens::WARNING
        } else {
            tokens::BORDER.with_alpha(180)
        })
        .stroke_width(if slot.selected { 2.5 } else { 1.0 })
        .radius(6.0)
        .shadow(if slot.selected {
            tokens::SHADOW_MD
        } else {
            0.0
        })
}

#[track_caller]
fn build_aetna_orientation_controls_view(label: String, is_rotated: bool) -> El {
    El::new(Kind::Custom("polychora_orientation_controls"))
        .at_loc(Location::caller())
        .style_profile(StyleProfile::Surface)
        .surface_role(SurfaceRole::Panel)
        .axis(Axis::Row)
        .children([
            column([
                row([
                    orientation_button("XZ", "xz", "Z key: rotate in XZ plane"),
                    orientation_button("YZ", "yz", "X key: rotate in YZ plane"),
                    orientation_button("XW", "xw", "C key: rotate in XW plane"),
                ])
                .gap(tokens::SPACE_1),
                row([
                    orientation_button("XY", "xy", "Rotate in XY plane"),
                    orientation_button("YW", "yw", "Rotate in YW plane"),
                    orientation_button("ZW", "zw", "Rotate in ZW plane"),
                ])
                .gap(tokens::SPACE_1),
            ])
            .gap(tokens::SPACE_1),
            column([
                button("Reset")
                    .key(format!("{ORIENTATION_KEY_PREFIX}reset"))
                    .tooltip("Reset orientation")
                    .secondary()
                    .width(Size::Fixed(60.0))
                    .height(Size::Fixed(24.0))
                    .padding(0.0),
                text(label)
                    .caption()
                    .center_text()
                    .width(Size::Fixed(60.0))
                    .color(if is_rotated {
                        Color::rgb(210, 196, 255)
                    } else {
                        tokens::MUTED_FOREGROUND
                    }),
            ])
            .gap(tokens::SPACE_1),
        ])
        .width(Size::Fixed(218.0))
        .height(Size::Hug)
        .padding(tokens::SPACE_2)
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .fill(if is_rotated {
            Color::rgba(42, 34, 76, 210)
        } else {
            tokens::CARD.with_alpha(205)
        })
        .stroke(tokens::BORDER.with_alpha(170))
        .radius(6.0)
        .shadow(tokens::SHADOW_MD)
}

fn short_label(name: String) -> String {
    if name.chars().count() > 14 {
        let prefix = name.chars().take(11).collect::<String>();
        format!("{prefix}...")
    } else {
        name
    }
}

fn orientation_button(label: &str, action: &str, tooltip: &str) -> El {
    button(label)
        .key(format!("{ORIENTATION_KEY_PREFIX}{action}"))
        .tooltip(tooltip)
        .ghost()
        .width(Size::Fixed(38.0))
        .height(Size::Fixed(24.0))
        .padding(0.0)
}

fn waila_panel(
    title: impl Into<String>,
    badge_label: impl Into<String>,
    detail_lines: impl IntoIterator<Item = String>,
) -> El {
    let details: Vec<El> = detail_lines
        .into_iter()
        .map(|line| mono(line).caption().muted().width(Size::Fill(1.0)))
        .collect();

    card([
        card_header([row([
            card_title(title).line_height(tokens::TEXT_BASE.size),
            spacer(),
            badge(badge_label).info(),
        ])
        .align(Align::Center)
        .gap(tokens::SPACE_3)]),
        card_content([column(details).gap(tokens::SPACE_1).width(Size::Fill(1.0))]).pt(0.0),
    ])
    .width(Size::Fixed(500.0))
    .height(Size::Hug)
    .axis(Axis::Column)
    .justify(Justify::Start)
    .shadow(tokens::SHADOW_LG)
}

impl App {
    pub(super) fn handle_aetna_ui_events(&mut self, events: Vec<aetna_core::UiEvent>) -> bool {
        let mut consumed = false;
        for event in events {
            let Some(route) = event.route() else {
                continue;
            };

            if let Some(slot_index) = aetna_hotbar_slot_index(route) {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.hotbar_selected_index = slot_index;
                    self.selected_block = block_data_from_slot(
                        self.inventory.hotbar_slot(self.hotbar_selected_index),
                    );
                }
            } else if let Some(action) = route.strip_prefix(ORIENTATION_KEY_PREFIX) {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.apply_aetna_orientation_action(action);
                }
            }
        }
        consumed
    }

    fn apply_aetna_orientation_action(&mut self, action: &str) {
        use polychora::shared::voxel::TesseractOrientation;

        self.placement_orientation = match action {
            "xz" => TesseractOrientation::ROT_XZ.compose(self.placement_orientation),
            "yz" => TesseractOrientation::ROT_YZ.compose(self.placement_orientation),
            "xw" => TesseractOrientation::ROT_XW.compose(self.placement_orientation),
            "xy" => TesseractOrientation::ROT_XY.compose(self.placement_orientation),
            "yw" => TesseractOrientation::ROT_YW.compose(self.placement_orientation),
            "zw" => TesseractOrientation::ROT_ZW.compose(self.placement_orientation),
            "reset" => TesseractOrientation::IDENTITY,
            _ => self.placement_orientation,
        };
    }
}

fn aetna_hotbar_slot_index(route: &str) -> Option<usize> {
    let suffix = route.strip_prefix(HOTBAR_SLOT_KEY_PREFIX)?;
    let index = suffix.parse::<usize>().ok()?;
    (index < 9).then_some(index)
}
