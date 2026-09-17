// Pixal3D model-set manifest の name<->role / model_family 契約。
// tools/model_manifest.py（model_files / infer_family / validate_manifest_shape の family 部分）と
// 同じ規則をブラウザ側でも判定する。両者の同一性は web/app/manifest_conformance.json を
// Python self-test と test_single_view.mjs の双方で読んで担保する。
//
// 揃えている規則（Python の shape 規則全体ではない。未知キー・型検査は各 validator 側）:
//   1. model_family が明示されていれば 'mv' | 'sv'
//   2. flow 4 role の name が全て `_mv.gguf` 末尾なら mv、全て `_sv.gguf` 末尾なら sv、混在・不明は null
//   3. 明示値と推定値が両方あって不一致なら拒否
//   4. family = 明示 || 推定 || 既定 'mv'
//   5. 既知 name は固定 role・required:true。未知 name に既知 role を付けるのも拒否
//      （未知 name + 未知 role の余剰は許容）
//   6. name 重複・role 重複を拒否
//   7. family の required 9 name / 9 role が全て揃う

export const MODEL_FAMILIES = ['mv', 'sv'];
export const DEFAULT_MODEL_FAMILY = 'mv';
const FLOW_ROLES = new Set(['ss_flow', 'shape_flow_512', 'shape_flow_1024', 'texture_flow_1024']);

// Python model_files(family) と 1 行ずつ対応させる（順序も同じ）。
export function modelFilesForFamily(family) {
  if (!MODEL_FAMILIES.includes(family)) throw new Error(`unknown model_family: ${String(family)}`);
  return {
    'dinov3.gguf': { role: 'image_encoder', required: true },
    'pixal3d_naf.gguf': { role: 'naf', required: true },
    [`pixal3d_ss_flow_${family}.gguf`]: { role: 'ss_flow', required: true },
    'ss_dec.gguf': { role: 'ss_decoder', required: true },
    [`pixal3d_shape_flow_512_${family}.gguf`]: { role: 'shape_flow_512', required: true },
    'shape_dec.gguf': { role: 'shape_decoder', required: true },
    [`pixal3d_shape_flow_1024_${family}.gguf`]: { role: 'shape_flow_1024', required: true },
    [`pixal3d_tex_flow_1024_${family}.gguf`]: { role: 'texture_flow_1024', required: true },
    'tex_dec.gguf': { role: 'texture_decoder', required: true },
  };
}

// Python infer_family: flow 4 role のファイル名末尾から family を推定する。混在・不明は null。
export function inferModelFamily(files) {
  if (!Array.isArray(files)) return null;
  const names = files
    .filter((ent) => ent && typeof ent === 'object' && FLOW_ROLES.has(ent.role) && typeof ent.name === 'string')
    .map((ent) => ent.name);
  if (!names.length) return null;
  for (const fam of MODEL_FAMILIES) {
    if (names.every((n) => n.endsWith(`_${fam}.gguf`))) return fam;
  }
  return null;
}

// Python validate_manifest_shape の family / name<->role / 完備部分。空配列なら合格。
export function manifestContractErrors(manifest) {
  const errors = [];
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) return ['manifest must be an object'];
  const files = manifest.files;
  if (!Array.isArray(files) || !files.length) return ['files must be a non-empty array'];

  let explicit = manifest.model_family;
  if (explicit !== undefined && explicit !== null && !MODEL_FAMILIES.includes(explicit)) {
    errors.push('model_family must be one of: ' + MODEL_FAMILIES.join(', '));
    explicit = null;
  }
  if (explicit === undefined) explicit = null;
  const inferred = inferModelFamily(files);
  if (explicit !== null && inferred !== null && explicit !== inferred) {
    errors.push(`model_family ${explicit} does not match the flow file names (${inferred})`);
  }
  const family = explicit || inferred || DEFAULT_MODEL_FAMILY;

  const expected = modelFilesForFamily(family);
  const roleOfName = new Map(Object.entries(expected).map(([name, spec]) => [name, spec.role]));
  const nameOfRole = new Map(Object.entries(expected).map(([name, spec]) => [spec.role, name]));
  const seenNames = new Set();
  const seenRoles = new Set();
  files.forEach((ent, i) => {
    const p = `files[${i}]`;
    if (!ent || typeof ent !== 'object') { errors.push(`${p} must be object`); return; }
    const name = ent.name;
    const role = ent.role;
    if (typeof name === 'string' && name) {
      if (seenNames.has(name)) errors.push(`duplicate file name: ${name}`);
      else seenNames.add(name);
    } else {
      errors.push(`${p}.name invalid`);
    }
    if (typeof role === 'string' && role) {
      if (seenRoles.has(role)) errors.push(`duplicate role: ${role}`);
      else seenRoles.add(role);
    } else {
      errors.push(`${p}.role invalid`);
    }
    // name と role の対応を固定する（role の入れ替えで別種の GGUF を別段へ渡させない）。
    if (typeof name === 'string' && typeof role === 'string') {
      if (roleOfName.has(name) && roleOfName.get(name) !== role) {
        errors.push(`${p}: ${name} must have role ${roleOfName.get(name)}, not ${role}`);
      } else if (!roleOfName.has(name) && nameOfRole.has(role)) {
        errors.push(`${p}: role ${role} must be ${nameOfRole.get(role)}, not ${name}`);
      }
      if (roleOfName.has(name) && ent.required !== true) {
        errors.push(`${p}: ${name} must be required`);
      }
    }
  });

  const missingNames = Object.entries(expected).filter(([name, spec]) => spec.required && !seenNames.has(name)).map(([name]) => name).sort();
  const missingRoles = [...new Set(Object.values(expected).filter((spec) => spec.required).map((spec) => spec.role))].filter((role) => !seenRoles.has(role)).sort();
  if (missingNames.length) errors.push('manifest missing required model files: ' + missingNames.join(', '));
  if (missingRoles.length) errors.push('manifest missing required roles: ' + missingRoles.join(', '));
  return errors;
}

// 契約違反が無ければ family（'mv' | 'sv'）、あれば null。null は UI では Generate 無効。
export function modelFamilyForManifest(manifest) {
  if (manifestContractErrors(manifest).length) return null;
  const explicit = manifest.model_family;
  if (explicit === 'mv' || explicit === 'sv') return explicit;
  return inferModelFamily(manifest.files) || DEFAULT_MODEL_FAMILY;
}
