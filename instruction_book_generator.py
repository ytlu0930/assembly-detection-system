"""Generate one long PNG assembly instruction sheet from V2 outputs."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "output" / "pipeline" / "error_aware_localization_smoke_test"
DEFAULT_NAME = "assembly_instruction_book.png"

PAGE_WIDTH = 1800
OUTER = 72
GAP = 40
PAD = 32
CARD_RADIUS = 24
MAX_IMAGE_H = 900

BG = "#F4F5F2"
CARD = "#FFFFFF"
TEXT = "#1F2933"
SUBTEXT = "#5B6770"
LINE = "#D8DEE3"
ACCENT = "#2474A6"
SUCCESS = "#2F855A"
WARNING = "#B7791F"
BADGE_BG = "#E9F3FA"
PLACEHOLDER = "#ECEFF1"

ERROR_ZH = {
    "correct": "組裝正確",
    "missingpart": "缺少零件",
    "extrapart": "多餘零件",
    "wrongpart": "錯誤零件",
    "positionerror": "位置錯誤",
    "criticalerror": "嚴重組裝錯誤",
    "uncertain": "結果不確定",
}

ACTION_ZH = {
    "prepare_part": "準備正確零件",
    "locate_installation_point": "確認安裝位置",
    "inspect_target": "確認目標區域",
    "insert_part": "安裝零件",
    "remove_part": "移除零件",
    "detach_part": "鬆開零件",
    "replace_part": "更換零件",
    "reposition_part": "調整零件位置",
    "reorient_part": "調整零件方向",
    "disassemble_local_area": "拆解局部結構",
    "rebuild_local_area": "重新組裝",
    "verify_local_result": "確認修正結果",
    "compare_reference": "與正確參考圖比對",
}


@dataclass
class ManualStep:
    sequence_index: int
    action: str
    title: str
    branch: str
    target_name: str
    instruction: str
    verification: str
    image_path: Optional[Path]
    output_filename: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class Metadata:
    model_id: str
    step_id: str
    image_name: str
    error_type: str
    requires_manual_review: bool


class InstructionBookGenerator:
    def __init__(
        self,
        *,
        page_width: int = PAGE_WIDTH,
        columns: int = 1,
        include_comparison: bool = True,
        include_warnings: bool = True,
    ) -> None:
        if page_width < 1000:
            raise ValueError("page_width must be at least 1000")
        if columns not in {1, 2}:
            raise ValueError("columns must be 1 or 2")

        self.page_width = page_width
        self.columns = columns
        self.include_comparison = include_comparison
        self.include_warnings = include_warnings

        self.font_title = self._font(54, bold=True)
        self.font_subtitle = self._font(30, bold=True)
        self.font_card_title = self._font(36, bold=True)
        self.font_body = self._font(27)
        self.font_body_bold = self._font(27, bold=True)
        self.font_small = self._font(22)
        self.font_badge = self._font(24, bold=True)

    def generate(
        self,
        prompts_json_path: str | Path,
        output_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        prompts_path = Path(prompts_json_path).expanduser().resolve()
        destination = Path(output_path).expanduser().resolve()

        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {destination}")

        package = self._load_json(prompts_path)
        metadata = Metadata(
            model_id=str(package.get("model_id", "")),
            step_id=str(package.get("step_id", "")),
            image_name=str(package.get("image_name", "")),
            error_type=str(package.get("overall_error_type", "uncertain")),
            requires_manual_review=bool(package.get("requires_manual_review", False)),
        )

        generated_root = prompts_path.parent / "generated_steps_v2"
        steps = self._parse_steps(package, generated_root)
        if not self.include_comparison:
            steps = [step for step in steps if step.branch != "composition"]

        canvas = self._render(metadata, steps)
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, format="PNG", optimize=True)
        return destination

    def _parse_steps(self, package: dict[str, Any], generated_root: Path) -> list[ManualStep]:
        raw_steps = package.get("step_prompts", [])
        if not isinstance(raw_steps, list):
            raise TypeError("step_prompts must be a list")

        result: list[ManualStep] = []
        for raw in raw_steps:
            if not isinstance(raw, dict):
                continue

            image_task = raw.get("image_task", {})
            if not isinstance(image_task, dict):
                image_task = {}

            structure = raw.get("instruction_structure", {})
            if not isinstance(structure, dict):
                structure = {}

            action = str(raw.get("action", "unknown"))
            branch = str(image_task.get("branch", "assembly"))
            output_filename = str(raw.get("output_filename", ""))
            target_name = str(
                raw.get(
                    "target_part_name_zh",
                    raw.get("target_part_visual_name_en", "目標零件"),
                )
            )

            result.append(
                ManualStep(
                    sequence_index=int(raw.get("sequence_index", 0)),
                    action=action,
                    title=str(raw.get("title", ACTION_ZH.get(action, action))),
                    branch=branch,
                    target_name=target_name,
                    instruction=self._instruction(raw, action, target_name),
                    verification=self._verification(action, target_name, structure),
                    image_path=self._resolve_image(generated_root, branch, output_filename),
                    output_filename=output_filename,
                    warnings=[str(x) for x in raw.get("warnings", [])]
                    if isinstance(raw.get("warnings", []), list)
                    else [],
                )
            )

        return sorted(result, key=lambda x: x.sequence_index)

    @staticmethod
    def _resolve_image(root: Path, branch: str, filename: str) -> Optional[Path]:
        folder = {
            "standalone": "standalone",
            "assembly": "assembly",
            "composition": "comparison",
        }.get(branch, branch)

        candidate = root / folder / filename
        if candidate.is_file():
            return candidate

        if filename:
            matches = list((root / folder).glob(f"{Path(filename).stem}.*")) if (root / folder).is_dir() else []
            if matches:
                return sorted(matches)[0]

        if branch == "composition":
            final_path = root / "final_comparison.png"
            if final_path.is_file():
                return final_path

        if branch == "assembly" and "verify" in filename:
            final_path = root / "final_assembly.png"
            if final_path.is_file():
                return final_path

        return None

    @staticmethod
    def _instruction(raw: dict[str, Any], action: str, target: str) -> str:
        if action == "prepare_part":
            return f"準備 1 個「{target}」，確認外型、顏色與孔位。"
        if action == "locate_installation_point":
            return f"依箭頭指示，確認「{target}」的正確安裝位置。"
        if action == "insert_part":
            pos = raw.get("expected_position_zh")
            ori = raw.get("expected_orientation_zh")
            detail = ""
            if pos or ori:
                items = []
                if pos:
                    items.append(f"位置：{pos}")
                if ori:
                    items.append(f"方向：{ori}")
                detail = f"（{'；'.join(items)}）"
            return f"將「{target}」安裝至指定位置{detail}。"
        if action == "remove_part":
            return f"依箭頭方向移除「{target}」。"
        if action == "replace_part":
            return f"將錯誤零件替換為「{target}」。"
        if action == "reposition_part":
            return f"將「{target}」移動到正確位置。"
        if action == "reorient_part":
            return f"將「{target}」旋轉至正確方向。"
        if action == "verify_local_result":
            return f"確認「{target}」已正確安裝，且周圍零件未位移。"
        if action == "compare_reference":
            return "將修正後模型與正確參考圖進行整體比對。"
        return f"依圖片完成「{target}」的修正操作。"

    @staticmethod
    def _verification(action: str, target: str, structure: dict[str, Any]) -> str:
        mapping = {
            "prepare_part": f"確認「{target}」的顏色、尺寸、形狀與孔位。",
            "locate_installation_point": "箭頭應只標示安裝位置，不應提前新增或移除零件。",
            "insert_part": "確認零件完全插入，且周圍零件沒有位移。",
            "remove_part": "確認目標零件已移除，其他正確零件仍維持原位。",
            "replace_part": "確認錯誤零件已替換為正確零件。",
            "reposition_part": "確認零件位置符合正確參考圖。",
            "reorient_part": "確認零件方向符合正確參考圖。",
            "verify_local_result": "確認修正區域的零件數量、位置、方向與顏色皆正確。",
            "compare_reference": "確認修正後模型與正確參考圖的整體結構一致。",
        }
        return mapping.get(action, str(structure.get("expected_result", "")))

    def _render(self, metadata: Metadata, steps: list[ManualStep]) -> Image.Image:
        header = self._header(metadata)
        footer = self._footer()

        content_width = self.page_width - OUTER * 2
        if self.columns == 1:
            cards = [self._card(step, content_width) for step in steps]
            total_h = OUTER + header.height + GAP + sum(c.height for c in cards)
            total_h += GAP * max(0, len(cards) - 1) + GAP + footer.height + OUTER
            canvas = Image.new("RGB", (self.page_width, total_h), BG)
            y = OUTER
            canvas.paste(header, (OUTER, y))
            y += header.height + GAP
            for i, card in enumerate(cards):
                canvas.paste(card, (OUTER, y))
                y += card.height + (GAP if i < len(cards) - 1 else 0)
            y += GAP
            canvas.paste(footer, (OUTER, y))
            return canvas

        col_gap = GAP
        col_width = (content_width - col_gap) // 2
        cards = [self._card(step, col_width) for step in steps]
        left: list[Image.Image] = []
        right: list[Image.Image] = []
        lh = rh = 0
        for card in cards:
            if lh <= rh:
                left.append(card)
                lh += card.height + GAP
            else:
                right.append(card)
                rh += card.height + GAP
        cards_h = max(max(0, lh - GAP), max(0, rh - GAP))
        total_h = OUTER + header.height + GAP + cards_h + GAP + footer.height + OUTER
        canvas = Image.new("RGB", (self.page_width, total_h), BG)
        canvas.paste(header, (OUTER, OUTER))
        y0 = OUTER + header.height + GAP
        y = y0
        for card in left:
            canvas.paste(card, (OUTER, y))
            y += card.height + GAP
        y = y0
        for card in right:
            canvas.paste(card, (OUTER + col_width + col_gap, y))
            y += card.height + GAP
        canvas.paste(footer, (OUTER, y0 + cards_h + GAP))
        return canvas

    def _header(self, metadata: Metadata) -> Image.Image:
        width = self.page_width - OUTER * 2
        height = 270
        image = Image.new("RGB", (width, height), CARD)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=CARD_RADIUS, fill=CARD, outline=LINE, width=2)
        draw.text((PAD, PAD), "AI 積木組裝修正說明書", fill=TEXT, font=self.font_title)
        error_zh = ERROR_ZH.get(metadata.error_type, metadata.error_type)
        draw.text(
            (PAD, 112),
            f"模型：{metadata.model_id}　步驟：{metadata.step_id}　錯誤類型：{error_zh}",
            fill=SUBTEXT,
            font=self.font_subtitle,
        )
        draw.text((PAD, 174), f"來源圖片：{metadata.image_name}", fill=SUBTEXT, font=self.font_small)
        badge = "需人工確認" if metadata.requires_manual_review else "自動流程"
        badge_color = WARNING if metadata.requires_manual_review else SUCCESS
        bbox = draw.textbbox((0, 0), badge, font=self.font_badge)
        bw = bbox[2] - bbox[0] + 34
        bh = bbox[3] - bbox[1] + 20
        bx = width - PAD - bw
        draw.rounded_rectangle((bx, PAD, bx + bw, PAD + bh), radius=18, fill=badge_color)
        draw.text((bx + 17, PAD + 8), badge, fill="white", font=self.font_badge)
        draw.line((PAD, height - 26, width - PAD, height - 26), fill=ACCENT, width=5)
        return image

    def _footer(self) -> Image.Image:
        width = self.page_width - OUTER * 2
        height = 155
        image = Image.new("RGB", (width, height), CARD)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=CARD_RADIUS, fill=CARD, outline=LINE, width=2)
        draw.text((PAD, PAD), "完成", fill=SUCCESS, font=self.font_card_title)
        draw.text((PAD, 88), "完成修正後，請重新拍照並再次執行 AI 檢測。", fill=TEXT, font=self.font_body)
        return image

    def _card(self, step: ManualStep, width: int) -> Image.Image:
        inner = width - PAD * 2
        step_image = self._image_or_placeholder(
            step.image_path,
            width=inner,
            max_height=MAX_IMAGE_H if width > 1000 else 620,
            label=f"找不到步驟圖片\n{step.output_filename}",
        )

        title = ACTION_ZH.get(step.action, step.title)
        title_lines = self._wrap(title, self.font_card_title, max(220, inner - 240))
        instruction_lines = self._wrap(step.instruction, self.font_body, inner)
        verification_lines = self._wrap(step.verification, self.font_small, inner)

        warning_lines: list[str] = []
        if self.include_warnings and step.warnings:
            warning_text = "；".join(self._warning_zh(w) for w in step.warnings)
            warning_lines = self._wrap(warning_text, self.font_small, inner)

        header_h = max(80, len(title_lines) * 50)
        instruction_h = max(44, len(instruction_lines) * 40)
        verification_h = 44 + len(verification_lines) * 32 if verification_lines else 0
        warning_h = 28 + len(warning_lines) * 31 if warning_lines else 0

        height = PAD + header_h + 18 + instruction_h + 22 + step_image.height + 20
        height += verification_h + warning_h + PAD

        card = Image.new("RGB", (width, height), CARD)
        draw = ImageDraw.Draw(card)
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=CARD_RADIUS, fill=CARD, outline=LINE, width=2)

        badge_text = f"STEP {step.sequence_index}"
        bb = draw.textbbox((0, 0), badge_text, font=self.font_badge)
        badge_w = bb[2] - bb[0] + 30
        draw.rounded_rectangle((PAD, PAD, PAD + badge_w, PAD + 46), radius=14, fill=BADGE_BG)
        draw.text((PAD + 15, PAD + 8), badge_text, fill=ACCENT, font=self.font_badge)

        tx = PAD + badge_w + 24
        ty = PAD
        for line in title_lines:
            draw.text((tx, ty), line, fill=TEXT, font=self.font_card_title)
            ty += 48

        y = PAD + header_h
        draw.line((PAD, y, width - PAD, y), fill=LINE, width=2)
        y += 20
        for line in instruction_lines:
            draw.text((PAD, y), line, fill=TEXT, font=self.font_body)
            y += 40

        y += 8
        card.paste(step_image, ((width - step_image.width) // 2, y))
        y += step_image.height + 18

        if verification_lines:
            draw.text((PAD, y), "檢查重點：", fill=SUCCESS, font=self.font_body_bold)
            y += 40
            for line in verification_lines:
                draw.text((PAD + 8, y), line, fill=SUBTEXT, font=self.font_small)
                y += 32

        if warning_lines:
            y += 10
            box_h = 24 + len(warning_lines) * 31
            draw.rounded_rectangle((PAD, y, width - PAD, y + box_h), radius=12, fill="#FFF7E6", outline="#E8C47A", width=1)
            wy = y + 10
            for line in warning_lines:
                draw.text((PAD + 16, wy), line, fill=WARNING, font=self.font_small)
                wy += 31

        return card

    def _image_or_placeholder(self, path: Optional[Path], *, width: int, max_height: int, label: str) -> Image.Image:
        if path and path.is_file():
            try:
                with Image.open(path) as raw:
                    image = raw.convert("RGB")
                image = ImageOps.contain(image, (width, max_height), method=Image.Resampling.LANCZOS)
                frame = Image.new("RGB", (width, image.height + 20), "#FAFAFA")
                frame.paste(image, ((width - image.width) // 2, 10))
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, width - 1, frame.height - 1), outline=LINE, width=2)
                return frame
            except Exception:
                pass

        height = min(500, max_height)
        image = Image.new("RGB", (width, height), PLACEHOLDER)
        draw = ImageDraw.Draw(image)
        lines = self._wrap(label, self.font_body, width - 80)
        y = (height - len(lines) * 42) // 2
        for line in lines:
            bb = draw.textbbox((0, 0), line, font=self.font_body)
            tw = bb[2] - bb[0]
            draw.text(((width - tw) // 2, y), line, fill=SUBTEXT, font=self.font_body)
            y += 42
        draw.rectangle((0, 0, width - 1, height - 1), outline=LINE, width=2)
        return image

    def _wrap(self, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
        if not text:
            return []
        lines: list[str] = []
        for paragraph in str(text).splitlines():
            paragraph = paragraph.strip()
            if not paragraph:
                lines.append("")
                continue
            current = ""
            for char in paragraph:
                candidate = current + char
                bb = font.getbbox(candidate)
                if bb[2] - bb[0] <= max_width:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = char
            if current:
                lines.append(current)
        return lines

    @staticmethod
    def _warning_zh(text: str) -> str:
        low = text.lower()
        if "manual review" in low:
            return "此步驟需要人工確認。"
        if "localization is unreliable" in low:
            return "定位可靠度較低，請確認箭頭或標示位置。"
        if "no bbox" in low:
            return "未取得可靠框選位置，請人工確認目標區域。"
        if "standalone" in low:
            return "此圖為獨立零件示意，不參與後續組裝圖串接。"
        if "python composition" in low:
            return "此比較圖由 Python 合成，不是 AI 重新生成。"
        if "assembly-branch start" in low:
            return "此步驟以原始錯誤照片作為起點。"
        return text

    @staticmethod
    def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
        regular = [
            "C:/Windows/Fonts/msjh.ttc",
            "C:/Windows/Fonts/mingliu.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        bolds = [
            "C:/Windows/Fonts/msjhbd.ttc",
            "C:/Windows/Fonts/msjh.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        for candidate in (bolds if bold else regular):
            path = Path(candidate)
            if path.is_file():
                try:
                    return ImageFont.truetype(str(path), size=size)
                except OSError:
                    continue
        return ImageFont.load_default()

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"JSON file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"Expected JSON object: {path}")
        return data


def find_latest_prompts_json() -> Path:
    files = sorted(
        DEFAULT_ROOT.glob("*/step_prompts_v2.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No step_prompts_v2.json found under:\n{DEFAULT_ROOT}")
    return files[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine all instruction steps into one long PNG image.")
    parser.add_argument("--prompts-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--columns", type=int, choices=[1, 2], default=1)
    parser.add_argument("--page-width", type=int, default=PAGE_WIDTH)
    parser.add_argument("--exclude-comparison", action="store_true")
    parser.add_argument("--exclude-warnings", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prompts_json = args.prompts_json.expanduser().resolve() if args.prompts_json else find_latest_prompts_json()
    output = args.output.expanduser().resolve() if args.output else prompts_json.parent / "generated_steps_v2" / DEFAULT_NAME

    generator = InstructionBookGenerator(
        page_width=args.page_width,
        columns=args.columns,
        include_comparison=not args.exclude_comparison,
        include_warnings=not args.exclude_warnings,
    )
    result = generator.generate(prompts_json, output, overwrite=args.overwrite)

    print("=" * 70)
    print("Instruction book generated")
    print("=" * 70)
    print(f"Source prompts: {prompts_json}")
    print(f"Output image:   {result}")
    print(f"Columns:        {args.columns}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise