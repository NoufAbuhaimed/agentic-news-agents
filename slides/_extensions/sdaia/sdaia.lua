-- SDAIA slides filter
-- Reads the `sdaia:` options block and turns shortcut classes into
-- reveal.js slide attributes.
--
-- Options (all optional):
--   sdaia:
--     mode: auto | light | dark     -- starting colour mode (default auto = follow the OS)
--     toggle: true | false          -- show the light/dark button and the D shortcut (default true)
--     accent: teal | coral | purple | navy   -- underline, links, progress bar (default teal)
--     title-background: true | false         -- animated pattern on the title slide (default true)
--     auto-sections: true | false   -- colour every level-1 heading as a section divider (default false)

local GRADIENTS = {
  teal   = "linear-gradient(135deg, #1C355E, #00C9A7)",
  coral  = "linear-gradient(135deg, #1C355E, #FF7A5C)",
  sunset = "linear-gradient(135deg, #FF7A5C, #1C355E)",
  purple = "linear-gradient(135deg, #1C355E, #9B8EC0)",
  navy   = "linear-gradient(135deg, #0F1F38, #1C355E)",
}
local CYCLE = { "teal", "coral", "sunset", "purple" }

local opts = {
  mode = "auto",
  toggle = true,
  accent = "teal",
  title_background = true,
  auto_sections = false,
}

local function str(v)
  if v == nil then return nil end
  return pandoc.utils.stringify(v)
end

local function bool(v, default)
  if v == nil then return default end
  if type(v) == "boolean" then return v end
  local s = str(v):lower()
  return not (s == "false" or s == "no" or s == "0")
end

local function has_class(el, c)
  for _, k in ipairs(el.classes) do if k == c then return true end end
  return false
end

local function add_class(el, c)
  if not has_class(el, c) then el.classes:insert(c) end
end

local function remove_class(el, c)
  local keep = pandoc.List()
  for _, k in ipairs(el.classes) do if k ~= c then keep:insert(k) end end
  el.classes = keep
end

local section_count = 0

local function read_meta(meta)
  local s = meta.sdaia
  if s ~= nil and type(s) == "table" then
    local m = str(s.mode)
    if m == "light" or m == "dark" or m == "auto" then opts.mode = m end
    local a = str(s.accent)
    if a == "teal" or a == "coral" or a == "purple" or a == "navy" then opts.accent = a end
    opts.toggle = bool(s.toggle, true)
    opts.title_background = bool(s["title-background"], true)
    opts.auto_sections = bool(s["auto-sections"], false)
  end

  if opts.title_background and meta["title-slide-attributes"] == nil then
    meta["title-slide-attributes"] = {
      ["data-background-image"] = pandoc.Inlines(quarto.utils.resolve_path("assets/anim.svg")),
      ["data-background-opacity"] = pandoc.Inlines("0.15"),
      ["data-background-size"] = pandoc.Inlines("cover"),
    }
  end

  quarto.doc.include_text("in-header", string.format(
    '<script>window.SDAIA_OPTIONS={mode:"%s",toggle:%s,accent:"%s"};' ..
    '(function(o){var h=document.documentElement;h.setAttribute("data-sdaia-accent",o.accent);' ..
    'var m=o.mode;try{var s=localStorage.getItem("sdaia-mode");if(o.toggle&&(s==="light"||s==="dark"))m=s;}catch(e){}' ..
    'if(m==="auto")m=window.matchMedia&&matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";' ..
    'if(m==="dark")h.classList.add("sdaia-mode-dark");})(window.SDAIA_OPTIONS);</script>',
    opts.mode, tostring(opts.toggle), opts.accent))
  return meta
end

local function header(el)
  if el.level > 2 then return nil end

  -- {.dark}: navy content slide
  if has_class(el, "dark") then
    remove_class(el, "dark")
    add_class(el, "sdaia-dark")
    if el.attributes["background-color"] == nil then
      el.attributes["background-color"] = "#1C355E"
    end
    return el
  end

  if el.level ~= 1 then return nil end

  -- {.section} or {.section-teal|coral|sunset|purple|navy}: gradient divider
  local chosen = nil
  for name, _ in pairs(GRADIENTS) do
    if has_class(el, "section-" .. name) then
      chosen = name
      remove_class(el, "section-" .. name)
    end
  end
  local explicit = has_class(el, "section")
  if explicit then remove_class(el, "section") end

  local is_section = chosen ~= nil or explicit or opts.auto_sections
  if has_class(el, "no-section") then
    remove_class(el, "no-section")
    is_section = false
  end
  if not is_section or el.attributes["background-gradient"] ~= nil
     or el.attributes["background-color"] ~= nil or el.attributes["background-image"] ~= nil then
    return el
  end

  section_count = section_count + 1
  chosen = chosen or CYCLE[((section_count - 1) % #CYCLE) + 1]
  el.attributes["background-gradient"] = GRADIENTS[chosen]
  add_class(el, "sdaia-dark")
  add_class(el, "sdaia-section")
  return el
end

return {
  { Meta = read_meta },
  { Header = header },
}
