use std::any::TypeId;
use std::fmt::{self, Debug, Formatter};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, LazyLock};

use comemo::{Track, Tracked};
use ecow::{EcoString, EcoVec, eco_format};
use hayagriva::archive::ArchivedStyle;
use hayagriva::io::BibLaTeXError;
use hayagriva::{
    BibliographyDriver, BibliographyRequest, CitationItem, CitationRequest, Library,
    SpecificLocator, TransparentLocator, citationberg,
};
use indexmap::IndexMap;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use smallvec::SmallVec;
use typst_syntax::{Span, Spanned, SyntaxMode};
use typst_utils::{ManuallyHash, NonZeroExt, PicoStr};

use crate::World;
use crate::diag::{
    At, HintedStrResult, LoadError, LoadResult, LoadedWithin, ReportPos,
    SourceDiagnostic, SourceResult, StrResult, bail, error, warning,
};
use crate::engine::{Engine, Sink};
use crate::foundations::{
    Bytes, Cast, CastInfo, Content, Context, Derived, FromValue, IntoValue, Label,
    NativeElement, OneOrMultiple, Packed, Reflect, Scope, ShowSet, Smart, StyleChain,
    Styles, Synthesize, Value, elem,
};
use crate::introspection::{
    EmptyIntrospector, History, Introspect, Introspector, Locatable, Location,
    QueryIntrospection,
};
use crate::layout::{BlockElem, Em, HElem, PadElem};
use crate::loading::{DataSource, Load, LoadSource, Loaded, format_yaml_error};
use crate::model::{
    CitationForm, CiteGroup, Destination, DirectLinkElem, FootnoteElem, HeadingElem,
    LinkElem, Url,
};
use crate::routines::SpanMode;
use crate::text::{Lang, LocalName, Region, SmallcapsElem, SubElem, SuperElem, TextElem};

/// A bibliography / reference listing.
///
/// You can create a new bibliography by calling this function with a path to a
/// bibliography file in either one of two formats:
///
/// - A Hayagriva `.yaml`/`.yml` file. Hayagriva is a new bibliography file
///   format designed for use with Typst. Visit its
///   #link("https://github.com/typst/hayagriva/blob/main/docs/file-format.md")[documentation]
///   for more details.
/// - A BibLaTeX `.bib` file.
///
/// By default, Typst processes citations with its built-in `{hayagriva}`
/// engine. The optional `engine` parameter can select an experimental Citum
/// backend with `{citum}`.
///
/// As soon as you add a bibliography somewhere in your document, you can start
/// citing things with reference syntax (`[@key]`) or explicit calls to the
/// @cite[citation] function (`[#cite(<key>)]`). The bibliography will only show
/// entries for works that were referenced in the document.
///
/// = Styles <styles>
/// Typst offers a wide selection of built-in
/// @bibliography.style[citation and bibliography styles]. Beyond those, you can
/// add and use custom #link("https://citationstyles.org/")[CSL] (Citation Style
/// Language) files. Wondering which style to use? Here are some good defaults
/// based on what discipline you're working in:
///
/// #docs-table(
///   table.header[Fields][Typical Styles],
///
///   [Engineering, IT],
///   [`{"ieee"}`],
///
///   [Psychology, Life Sciences],
///   [`{"apa"}`],
///
///   [Social sciences],
///   [`{"chicago-author-date"}`],
///
///   [Humanities],
///   [`{"mla"}`, `{"chicago-notes"}`, `{"harvard-cite-them-right"}`],
///
///   [Economics],
///   [`{"harvard-cite-them-right"}`],
///
///   [Physics],
///   [`{"american-physics-society"}`],
/// )
///
/// = Example <example>
/// ```example
/// This was already noted by
/// pirates long ago. @arrgh
///
/// Multiple sources say ...
/// @arrgh @netwok.
///
/// #bibliography("works.bib")
/// ```
#[elem(Locatable, Synthesize, ShowSet, LocalName)]
pub struct BibliographyElem {
    /// One or multiple paths to or raw bytes for Hayagriva `.yaml` and/or
    /// BibLaTeX `.bib` files.
    ///
    /// This can be a:
    /// - A path string or @path to load a bibliography file from.
    /// - Raw bytes from which the bibliography should be decoded.
    /// - An array where each item is one of the above.
    #[required]
    #[parse(
        let citation_engine = args.named("engine")?.unwrap_or_default();
        let sources: Spanned<OneOrMultiple<DataSource>> = args.expect("sources")?;
        let citum_style = match citation_engine {
            CitationEngine::Hayagriva => None,
            CitationEngine::Citum => match args.named::<Spanned<DataSource>>("style")? {
                Some(source) => Some(source),
                None => bail!(sources.span, "citation engine \"citum\" requires a style"),
            },
        };
        Bibliography::load(engine.world, sources, citation_engine, citum_style)?
    )]
    pub sources: Derived<OneOrMultiple<DataSource>, Bibliography>,

    /// The citation engine to use.
    ///
    /// The default engine is `{hayagriva}`, which is Typst's built-in citation
    /// processor. The `{citum}` engine is experimental and currently renders
    /// Citum's plain-text output as Typst text.
    #[external]
    #[default(CitationEngine::Hayagriva)]
    pub engine: CitationEngine,

    /// The title of the bibliography.
    ///
    /// - When set to `{auto}`, an appropriate title for the
    ///   @text.lang[text language] will be used. This is the default.
    /// - When set to `{none}`, the bibliography will not have a title.
    /// - A custom title can be set by passing content.
    ///
    /// The bibliography's heading will not be numbered by default, but you can
    /// force it to be with a show-set rule:
    /// `{show bibliography: set heading(numbering: "1.")}`
    pub title: Smart<Option<Content>>,

    /// Whether to include all works from the given bibliography files, even
    /// those that weren't cited in the document.
    ///
    /// To selectively add individual cited works without showing them, you can
    /// also use the `cite` function with @cite.form[`form`] set to `{none}`.
    #[default(false)]
    pub full: bool,

    /// The bibliography style.
    ///
    /// This can be:
    /// - A string with the name of one of the built-in styles (see below). Some
    ///   of the styles listed below appear twice, once with their full name and
    ///   once with a short alias.
    /// - A path string or @path to a
    ///   #link("https://citationstyles.org/")[CSL file].
    /// - Raw bytes from which a CSL style should be decoded.
    #[parse(match args.named::<Spanned<CslSource>>("style")? {
        Some(source) => Some(CslStyle::load(engine, source)?),
        None => None,
    })]
    #[default({
        let default = ArchivedStyle::InstituteOfElectricalAndElectronicsEngineers;
        Derived::new(CslSource::Named(default, None), CslStyle::from_archived(default))
    })]
    pub style: Derived<CslSource, CslStyle>,

    /// The language setting where the bibliography is.
    #[internal]
    #[synthesized]
    pub lang: Lang,

    /// The region setting where the bibliography is.
    #[internal]
    #[synthesized]
    pub region: Option<Region>,
}

/// The citation engine used for processing citations and bibliographies.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Cast)]
pub enum CitationEngine {
    /// Typst's built-in citation engine.
    #[default]
    Hayagriva,
    /// Reserved for future Citum integration.
    Citum,
}

impl BibliographyElem {
    /// Find the document's bibliography.
    pub fn find(engine: &mut Engine, span: Span) -> StrResult<Packed<Self>> {
        let elems = engine.introspect(QueryIntrospection(Self::ELEM.select(), span));

        let mut iter = elems.iter();
        let Some(elem) = iter.next() else {
            bail!("the document does not contain a bibliography");
        };

        if iter.next().is_some() {
            bail!("multiple bibliographies are not yet supported");
        }

        Ok(elem.to_packed::<Self>().unwrap().clone())
    }

    /// Whether the bibliography contains the given key.
    pub fn has(engine: &mut Engine, key: Label, span: Span) -> bool {
        engine
            .introspect(QueryIntrospection(Self::ELEM.select(), span))
            .iter()
            .any(|elem| elem.to_packed::<Self>().unwrap().sources.derived.has(key))
    }

    /// Find all bibliography keys.
    pub fn keys(
        introspector: Tracked<dyn Introspector + '_>,
    ) -> Vec<(Label, Option<EcoString>)> {
        let mut vec = vec![];
        for elem in introspector.query(&Self::ELEM.select()).iter() {
            let this = elem.to_packed::<Self>().unwrap();
            vec.extend(this.sources.derived.keys());
        }
        vec
    }
}

impl Packed<BibliographyElem> {
    /// Produces the heading for the bibliography, if any.
    pub fn realize_title(&self, styles: StyleChain) -> Option<Content> {
        self.title
            .get_cloned(styles)
            .unwrap_or_else(|| {
                Some(TextElem::packed(Packed::<BibliographyElem>::local_name_in(styles)))
            })
            .map(|title| {
                HeadingElem::new(title)
                    .with_depth(NonZeroUsize::ONE)
                    .pack()
                    .spanned(self.span())
            })
    }
}

impl Synthesize for Packed<BibliographyElem> {
    fn synthesize(&mut self, _: &mut Engine, styles: StyleChain) -> SourceResult<()> {
        let elem = self.as_mut();
        elem.lang = Some(styles.get(TextElem::lang));
        elem.region = Some(styles.get(TextElem::region));
        Ok(())
    }
}

impl ShowSet for Packed<BibliographyElem> {
    fn show_set(&self, _: StyleChain) -> Styles {
        const INDENT: Em = Em::new(1.0);
        let mut out = Styles::new();
        out.set(HeadingElem::numbering, None);
        out.set(PadElem::left, INDENT.into());
        out
    }
}

impl LocalName for Packed<BibliographyElem> {
    const KEY: &'static str = "bibliography";
}

/// A loaded bibliography for one of Typst's citation engines.
#[derive(Clone, PartialEq, Hash)]
pub enum Bibliography {
    /// Bibliography data decoded for the built-in Hayagriva engine.
    Hayagriva(HayagrivaBibliography),
    /// Bibliography data decoded for the experimental Citum engine.
    Citum(CitumBibliography),
}

impl Bibliography {
    /// Load a bibliography from data sources.
    fn load(
        world: Tracked<dyn World + '_>,
        sources: Spanned<OneOrMultiple<DataSource>>,
        engine: CitationEngine,
        citum_style: Option<Spanned<DataSource>>,
    ) -> SourceResult<Derived<OneOrMultiple<DataSource>, Self>> {
        let bibliography = match engine {
            CitationEngine::Hayagriva => {
                let loaded = sources.load(world)?;
                Self::Hayagriva(HayagrivaBibliography::decode(&loaded)?)
            }
            CitationEngine::Citum => {
                let loaded = sources.load(world)?;
                let style_source = citum_style.expect("Citum style should be parsed");
                let style = style_source.load(world)?;
                Self::Citum(CitumBibliography::decode(&loaded, &style)?)
            }
        };
        Ok(Derived::new(sources.v, bibliography))
    }

    fn engine(&self) -> CitationEngine {
        match self {
            Self::Hayagriva(..) => CitationEngine::Hayagriva,
            Self::Citum(..) => CitationEngine::Citum,
        }
    }

    fn as_hayagriva(&self) -> &HayagrivaBibliography {
        match self {
            Self::Hayagriva(bibliography) => bibliography,
            Self::Citum(..) => unreachable!("expected Hayagriva bibliography"),
        }
    }

    fn as_citum(&self) -> &CitumBibliography {
        match self {
            Self::Citum(bibliography) => bibliography,
            Self::Hayagriva(..) => unreachable!("expected Citum bibliography"),
        }
    }

    fn has(&self, key: Label) -> bool {
        match self {
            Self::Hayagriva(bibliography) => bibliography.has(key),
            Self::Citum(bibliography) => bibliography.has(key),
        }
    }

    fn keys(&self) -> Vec<(Label, Option<EcoString>)> {
        match self {
            Self::Hayagriva(bibliography) => bibliography
                .iter()
                .map(|(key, entry)| {
                    let detail = entry.title().map(|title| title.value.to_str().into());
                    (key, detail)
                })
                .collect(),
            Self::Citum(bibliography) => bibliography
                .iter_keys()
                .filter_map(|key| {
                    Label::new(PicoStr::intern(key)).map(|label| (label, None))
                })
                .collect(),
        }
    }
}

impl Debug for Bibliography {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::Hayagriva(bibliography) => {
                f.debug_tuple("Hayagriva").field(bibliography).finish()
            }
            Self::Citum(bibliography) => {
                f.debug_tuple("Citum").field(bibliography).finish()
            }
        }
    }
}

/// Bibliography data decoded for Hayagriva.
#[derive(Clone, PartialEq, Hash)]
pub struct HayagrivaBibliography(
    Arc<ManuallyHash<IndexMap<Label, hayagriva::Entry, FxBuildHasher>>>,
);

impl HayagrivaBibliography {
    /// Decode a bibliography from loaded data sources.
    #[comemo::memoize]
    #[typst_macros::time(name = "load bibliography")]
    fn decode(data: &[Loaded]) -> SourceResult<HayagrivaBibliography> {
        let mut map = IndexMap::default();
        let mut duplicates = Vec::<EcoString>::new();

        // We might have multiple bib/yaml files
        for d in data.iter() {
            let library = decode_library(d)?;
            for entry in library {
                let label = Label::new(PicoStr::intern(entry.key()))
                    .ok_or("bibliography contains entry with empty key")
                    .at(d.source.span)?;

                match map.entry(label) {
                    indexmap::map::Entry::Vacant(vacant) => {
                        vacant.insert(entry);
                    }
                    indexmap::map::Entry::Occupied(_) => {
                        duplicates.push(entry.key().into());
                    }
                }
            }
        }

        if !duplicates.is_empty() {
            // TODO: Store spans of entries for duplicate key error messages.
            // Requires hayagriva entries to store their location, which should
            // be fine, since they are 1kb anyway.
            let span = data.first().unwrap().source.span;
            bail!(span, "duplicate bibliography keys: {}", duplicates.join(", "));
        }

        Ok(HayagrivaBibliography(Arc::new(ManuallyHash::new(
            map,
            typst_utils::hash128(data),
        ))))
    }

    fn has(&self, key: Label) -> bool {
        self.0.contains_key(&key)
    }

    fn get(&self, key: Label) -> Option<&hayagriva::Entry> {
        self.0.get(&key)
    }

    fn iter(&self) -> impl Iterator<Item = (Label, &hayagriva::Entry)> {
        self.0.iter().map(|(&k, v)| (k, v))
    }
}

impl Debug for HayagrivaBibliography {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_set().entries(self.0.keys()).finish()
    }
}

/// Bibliography data decoded for Citum.
#[derive(Clone, PartialEq, Hash)]
pub struct CitumBibliography(Arc<ManuallyHash<CitumBibliographyData>>);

/// Loaded Citum bibliography data and style.
#[derive(Debug)]
struct CitumBibliographyData {
    references: citum_engine::Bibliography,
    style: citum_engine::Style,
}

impl CitumBibliography {
    /// Decode a bibliography from loaded data sources and a loaded Citum style.
    #[comemo::memoize]
    #[typst_macros::time(name = "load citum bibliography")]
    fn decode(data: &[Loaded], style: &Loaded) -> SourceResult<CitumBibliography> {
        let mut references = citum_engine::Bibliography::new();
        let mut duplicates = Vec::<EcoString>::new();

        for loaded in data {
            for (key, reference) in decode_citum_bibliography(loaded)? {
                if references.insert(key.clone(), reference).is_some() {
                    duplicates.push(key.into());
                }
            }
        }

        if !duplicates.is_empty() {
            let span = data.first().unwrap().source.span;
            bail!(span, "duplicate bibliography keys: {}", duplicates.join(", "));
        }

        let style_data = citum_engine::Style::from_yaml_bytes(style.data.as_slice())
            .map_err(|err| {
                LoadError::new(ReportPos::None, "failed to load Citum style", err)
            })
            .within(style)?;

        Ok(CitumBibliography(Arc::new(ManuallyHash::new(
            CitumBibliographyData { references, style: style_data },
            typst_utils::hash128(&(data, style)),
        ))))
    }

    fn has(&self, key: Label) -> bool {
        self.0.references.contains_key(key.resolve().as_str())
    }

    fn iter_keys(&self) -> impl Iterator<Item = &str> {
        self.0.references.keys().map(String::as_str)
    }

    fn references(&self) -> &citum_engine::Bibliography {
        &self.0.references
    }

    fn style(&self) -> &citum_engine::Style {
        &self.0.style
    }

    fn is_note_style(&self) -> bool {
        matches!(
            self.style().options.as_ref().map(|options| &options.processing),
            Some(Some(citum_engine::Processing::Note)),
        )
    }
}

impl Debug for CitumBibliography {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_set().entries(self.0.references.keys()).finish()
    }
}

/// Decode Citum bibliography data from one data source.
fn decode_citum_bibliography(
    loaded: &Loaded,
) -> SourceResult<citum_engine::Bibliography> {
    let bytes = loaded.data.as_slice();

    if let LoadSource::Path(file_id) = loaded.source.v {
        let ext = file_id.vpath().extension().unwrap_or_default();
        match ext.to_lowercase().as_str() {
            "json" => serde_json::from_slice(bytes)
                .map_err(format_citum_json_error)
                .within(loaded),
            "yml" | "yaml" => serde_yaml::from_slice(bytes)
                .map_err(format_citum_yaml_error)
                .within(loaded),
            _ => bail!(
                loaded.source.span,
                "unknown Citum bibliography format (must be .json, .yaml, or .yml)"
            ),
        }
    } else {
        match serde_json::from_slice(bytes) {
            Ok(references) => Ok(references),
            Err(json_err) => serde_yaml::from_slice(bytes)
                .map_err(|yaml_err| {
                    LoadError::new(
                        ReportPos::None,
                        "failed to parse Citum bibliography",
                        format!(
                            "not valid Citum JSON ({json_err}) or Citum YAML ({yaml_err})"
                        ),
                    )
                })
                .within(loaded),
        }
    }
}

/// Format a Citum JSON loading error.
fn format_citum_json_error(error: serde_json::Error) -> LoadError {
    LoadError::new(ReportPos::None, "failed to parse Citum JSON", error)
}

/// Format a Citum YAML loading error.
fn format_citum_yaml_error(error: serde_yaml::Error) -> LoadError {
    LoadError::new(ReportPos::None, "failed to parse Citum YAML", error)
}

/// Decode on library from one data source.
fn decode_library(loaded: &Loaded) -> SourceResult<Library> {
    let data = loaded.data.as_str().within(loaded)?;

    if let LoadSource::Path(file_id) = loaded.source.v {
        // If we got a path, use the extension to determine whether it is
        // YAML or BibLaTeX.
        let ext = file_id.vpath().extension().unwrap_or_default();
        match ext.to_lowercase().as_str() {
            "yml" | "yaml" => hayagriva::io::from_yaml_str(data)
                .map_err(format_yaml_error)
                .within(loaded),
            "bib" => hayagriva::io::from_biblatex_str(data)
                .map_err(format_biblatex_error)
                .within(loaded),
            _ => bail!(
                loaded.source.span,
                "unknown bibliography format (must be .yaml/.yml or .bib)"
            ),
        }
    } else {
        // If we just got bytes, we need to guess. If it can be decoded as
        // hayagriva YAML, we'll use that.
        let haya_err = match hayagriva::io::from_yaml_str(data) {
            Ok(library) => return Ok(library),
            Err(err) => err,
        };

        // If it can be decoded as BibLaTeX, we use that instead.
        let bib_errs = match hayagriva::io::from_biblatex_str(data) {
            // If the file is almost valid yaml, but contains no `@` character
            // it will be successfully parsed as an empty BibLaTeX library,
            // since BibLaTeX does support arbitrary text outside of entries.
            Ok(library) if !library.is_empty() => return Ok(library),
            Ok(_) => None,
            Err(err) => Some(err),
        };

        // If neither decoded correctly, check whether `:` or `{` appears
        // more often to guess whether it's more likely to be YAML or BibLaTeX
        // and emit the more appropriate error.
        let mut yaml = 0;
        let mut biblatex = 0;
        for c in data.chars() {
            match c {
                ':' => yaml += 1,
                '{' => biblatex += 1,
                _ => {}
            }
        }

        match bib_errs {
            Some(bib_errs) if biblatex >= yaml => {
                Err(format_biblatex_error(bib_errs)).within(loaded)
            }
            _ => Err(format_yaml_error(haya_err)).within(loaded),
        }
    }
}

/// Format a BibLaTeX loading error.
fn format_biblatex_error(errors: Vec<BibLaTeXError>) -> LoadError {
    // TODO: return multiple errors?
    let Some(error) = errors.into_iter().next() else {
        // TODO: can this even happen, should we just unwrap?
        return LoadError::new(
            ReportPos::None,
            "failed to parse BibLaTeX",
            "something went wrong",
        );
    };

    let (range, msg) = match error {
        BibLaTeXError::Parse(error) => (error.span, error.kind.to_string()),
        BibLaTeXError::Type(error) => (error.span, error.kind.to_string()),
    };

    LoadError::new(range, "failed to parse BibLaTeX", msg)
}

/// A loaded CSL style.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct CslStyle(Arc<ManuallyHash<citationberg::IndependentStyle>>);

impl CslStyle {
    /// Load a CSL style from a data source.
    pub fn load(
        engine: &mut Engine,
        Spanned { v: source, span }: Spanned<CslSource>,
    ) -> SourceResult<Derived<CslSource, Self>> {
        let style = match &source {
            CslSource::Named(style, deprecation) => {
                if let Some(message) = deprecation {
                    engine.sink.warn(warning!(span, "{message}"));
                }
                Self::from_archived(*style)
            }
            CslSource::Normal(source) => {
                let loaded = Spanned::new(source, span).load(engine.world)?;
                Self::from_data(&loaded.data).within(&loaded)?
            }
        };
        Ok(Derived::new(source, style))
    }

    /// Load a built-in CSL style.
    #[comemo::memoize]
    pub fn from_archived(archived: ArchivedStyle) -> CslStyle {
        match archived.get() {
            citationberg::Style::Independent(style) => Self(Arc::new(ManuallyHash::new(
                style,
                typst_utils::hash128(&(TypeId::of::<ArchivedStyle>(), archived)),
            ))),
            // Ensured by `test_bibliography_load_builtin_styles`.
            _ => unreachable!("archive should not contain dependent styles"),
        }
    }

    /// Load a CSL style from file contents.
    #[comemo::memoize]
    pub fn from_data(bytes: &Bytes) -> LoadResult<CslStyle> {
        let text = bytes.as_str()?;
        citationberg::IndependentStyle::from_xml(text)
            .map(|style| {
                Self(Arc::new(ManuallyHash::new(
                    style,
                    typst_utils::hash128(&(TypeId::of::<Bytes>(), bytes)),
                )))
            })
            .map_err(|err| {
                LoadError::new(ReportPos::None, "failed to load CSL style", err)
            })
    }

    /// Get the underlying independent style.
    pub fn get(&self) -> &citationberg::IndependentStyle {
        self.0.as_ref()
    }
}

/// Source for a CSL style.
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum CslSource {
    /// A predefined named style and potentially a deprecation warning.
    Named(ArchivedStyle, Option<&'static str>),
    /// A normal data source.
    Normal(DataSource),
}

impl Reflect for CslSource {
    #[comemo::memoize]
    fn input() -> CastInfo {
        let source = std::iter::once(DataSource::input());

        /// All possible names and their short documentation for `ArchivedStyle`, including aliases.
        static ARCHIVED_STYLE_NAMES: LazyLock<Vec<(&&str, &'static str)>> =
            LazyLock::new(|| {
                ArchivedStyle::all()
                    .iter()
                    .flat_map(|name| {
                        let (main_name, aliases) = name
                            .names()
                            .split_first()
                            .expect("all ArchivedStyle should have at least one name");

                        std::iter::once((main_name, name.display_name())).chain(
                            aliases.iter().map(move |alias| {
                                // Leaking is okay here, because we are in a `LazyLock`.
                                let docs: &'static str = Box::leak(
                                    format!("A short alias of `{main_name}`")
                                        .into_boxed_str(),
                                );
                                (alias, docs)
                            }),
                        )
                    })
                    .collect()
            });
        let names = ARCHIVED_STYLE_NAMES
            .iter()
            .map(|(value, docs)| CastInfo::Value(value.into_value(), docs));

        CastInfo::Union(source.into_iter().chain(names).collect())
    }

    fn output() -> CastInfo {
        DataSource::output()
    }

    fn castable(value: &Value) -> bool {
        DataSource::castable(value)
    }
}

impl FromValue for CslSource {
    fn from_value(value: Value) -> HintedStrResult<Self> {
        if EcoString::castable(&value) {
            let string = EcoString::from_value(value.clone())?;
            if Path::new(string.as_str()).extension().is_none() {
                let mut warning = None;
                if string.as_str() == "chicago-fullnotes" {
                    warning = Some(
                        "style \"chicago-fullnotes\" has been deprecated \
                         in favor of \"chicago-notes\"",
                    );
                } else if string.as_str() == "modern-humanities-research-association" {
                    warning = Some(
                        "style \"modern-humanities-research-association\" \
                         has been deprecated in favor of \
                         \"modern-humanities-research-association-notes\"",
                    );
                }

                let style = ArchivedStyle::by_name(&string)
                    .ok_or_else(|| eco_format!("unknown style: {}", string))?;
                return Ok(CslSource::Named(style, warning));
            }
        }

        DataSource::from_value(value).map(CslSource::Normal)
    }
}

impl IntoValue for CslSource {
    fn into_value(self) -> Value {
        match self {
            // We prefer the shorter names which are at the back of the array.
            Self::Named(v, _) => v.names().last().unwrap().into_value(),
            Self::Normal(v) => v.into_value(),
        }
    }
}

/// Fully formatted citations and references, generated once (through
/// memoization) for the whole document. This setup is necessary because
/// citation formatting is inherently stateful and we need access to all
/// citations to do it.
pub struct Works {
    /// Maps from the location of a citation group to its rendered content.
    pub citations: FxHashMap<Location, SourceResult<Content>>,
    /// Lists all references in the bibliography, with optional prefix, or
    /// `None` if the citation style can't be used for bibliographies.
    pub references: Option<Vec<(Option<Content>, Content, Location)>>,
    /// Whether the bibliography should have hanging indent.
    pub hanging_indent: bool,
}

impl Works {
    /// Generate all citations and the whole bibliography.
    pub fn generate(engine: &mut Engine, span: Span) -> SourceResult<Arc<Works>> {
        let bibliography = BibliographyElem::find(engine, span).at(span)?;
        let groups = engine.introspect(CiteGroupIntrospection(span));
        Self::generate_impl(engine.world, bibliography, groups).at(span)
    }

    /// Generate all citations and the whole bibliography, given an existing
    /// bibliography (no need to query it).
    pub fn with_bibliography(
        engine: &mut Engine,
        bibliography: Packed<BibliographyElem>,
    ) -> SourceResult<Arc<Works>> {
        let span = bibliography.span();
        let groups = engine.introspect(CiteGroupIntrospection(span));
        Self::generate_impl(engine.world, bibliography, groups).at(span)
    }

    /// The internal implementation of [`Works::generate`].
    #[comemo::memoize]
    fn generate_impl(
        world: Tracked<dyn World + '_>,
        bibliography: Packed<BibliographyElem>,
        groups: EcoVec<Content>,
    ) -> StrResult<Arc<Works>> {
        match bibliography.sources.derived.engine() {
            CitationEngine::Hayagriva => {
                HayagrivaBackend::generate(world, bibliography, groups)
            }
            CitationEngine::Citum => CitumBackend::generate(world, bibliography, groups),
        }
    }

    /// Extracts the generated references, failing with an error if none have
    /// been generated.
    pub fn references<'a>(
        &'a self,
        elem: &Packed<BibliographyElem>,
        styles: StyleChain,
    ) -> SourceResult<&'a [(Option<Content>, Content, Location)]> {
        self.references
            .as_deref()
            .ok_or_else(|| match elem.style.get_ref(styles).source {
                CslSource::Named(style, _) => eco_format!(
                    "CSL style \"{}\" is not suitable for bibliographies",
                    style.display_name()
                ),
                CslSource::Normal(..) => {
                    "CSL style is not suitable for bibliographies".into()
                }
            })
            .at(elem.span())
    }
}

/// Retrieves all citation groups in the document.
///
/// This is separate from `QueryIntrospection` so that we can customize the
/// diagnostic as the `CiteGroup` is internal. The default query message is also
/// not that helpful in this case.
#[derive(Debug, Clone, PartialEq, Hash)]
struct CiteGroupIntrospection(Span);

impl Introspect for CiteGroupIntrospection {
    type Output = EcoVec<Content>;

    fn introspect(
        &self,
        _: &mut Engine,
        introspector: Tracked<dyn Introspector + '_>,
    ) -> Self::Output {
        introspector.query(&CiteGroup::ELEM.select())
    }

    fn diagnose(&self, _: &History<Self::Output>) -> SourceDiagnostic {
        warning!(
            self.0, "citation grouping did not stabilize";
            hint: "this can happen if the citations and bibliographies in the \
                   document did not stabilize by the end of the third layout iteration";
        )
    }
}

/// Backend interface for citation engines.
trait CitationEngineBackend {
    /// Generate all citations and bibliography entries for the backend.
    fn generate(
        world: Tracked<dyn World + '_>,
        bibliography: Packed<BibliographyElem>,
        groups: EcoVec<Content>,
    ) -> StrResult<Arc<Works>>;
}

/// Hayagriva citation engine backend.
struct HayagrivaBackend;

impl CitationEngineBackend for HayagrivaBackend {
    fn generate(
        world: Tracked<dyn World + '_>,
        bibliography: Packed<BibliographyElem>,
        groups: EcoVec<Content>,
    ) -> StrResult<Arc<Works>> {
        let mut generator = HayagrivaGenerator::new(world, bibliography, groups)?;
        let rendered = generator.drive();
        let works = generator.display(&rendered)?;
        Ok(Arc::new(works))
    }
}

/// Citum citation engine backend.
struct CitumBackend;

impl CitationEngineBackend for CitumBackend {
    fn generate(
        _world: Tracked<dyn World + '_>,
        bibliography: Packed<BibliographyElem>,
        groups: EcoVec<Content>,
    ) -> StrResult<Arc<Works>> {
        let database = bibliography.sources.derived.as_citum().clone();
        let mut generator = CitumGenerator::new(database, bibliography, groups);
        let works = generator.generate()?;
        Ok(Arc::new(works))
    }
}

/// Context for generating the bibliography with Citum.
struct CitumGenerator {
    /// The document's Citum bibliography.
    database: CitumBibliography,
    /// The document's bibliography element.
    bibliography: Packed<BibliographyElem>,
    /// The document's citation groups.
    groups: EcoVec<Content>,
    /// Details about each citation group in document order.
    infos: Vec<CitumGroupInfo>,
    /// Citations with unresolved keys or unsupported options.
    failures: FxHashMap<Location, SourceResult<Content>>,
    /// Cited keys in document order.
    cited: Vec<String>,
}

/// Details about a Citum citation group.
struct CitumGroupInfo {
    /// The group's location.
    location: Location,
    /// The group's span.
    span: Span,
    /// Whether all citations in the group were hidden.
    hidden: bool,
}

impl CitumGenerator {
    /// Create a new Citum generator.
    fn new(
        database: CitumBibliography,
        bibliography: Packed<BibliographyElem>,
        groups: EcoVec<Content>,
    ) -> Self {
        Self {
            database,
            bibliography,
            groups,
            infos: Vec::new(),
            failures: FxHashMap::default(),
            cited: Vec::new(),
        }
    }

    /// Generate all Citum citations and references.
    fn generate(&mut self) -> StrResult<Works> {
        let citations = self.collect_citations();
        let processor = citum_engine::Processor::new(
            self.database.style().clone(),
            self.database.references().clone(),
        );

        let rendered = processor
            .process_citations_with_format::<citum_engine::render::plain::PlainText>(
                &citations,
            )
            .map_err(|err| eco_format!("failed to process Citum citations: {err}"))?;

        let citations = self.display_citations(&rendered);
        let references = self.display_references(&processor);
        Ok(Works {
            citations,
            references: Some(references),
            hanging_indent: false,
        })
    }

    /// Convert Typst citation groups into Citum citation requests.
    fn collect_citations(&mut self) -> Vec<citum_engine::Citation> {
        let mut citations = Vec::new();

        for (index, elem) in self.groups.iter().enumerate() {
            let group = elem.to_packed::<CiteGroup>().unwrap();
            let location = elem.location().unwrap();
            let children = &group.children;
            let Some(first) = children.first() else { continue };

            let mut items = Vec::with_capacity(children.len());
            let mut errors = EcoVec::new();
            let mut hidden = true;
            let mut prose = false;
            let mut normal = false;

            for child in children {
                if matches!(child.style.get_ref(StyleChain::default()), Smart::Custom(_))
                {
                    errors.push(error!(
                        child.span(),
                        "per-citation style overrides are not supported by citation engine \"citum\"",
                    ));
                    continue;
                }

                let key = child.key.resolve().to_string();
                if !self.database.references().contains_key(&key) {
                    errors.push(error!(
                        child.span(),
                        "key `{}` does not exist in the bibliography",
                        child.key.resolve(),
                    ));
                    continue;
                }

                let form = child.form.get(StyleChain::default());
                let citation_hidden = form.is_none();
                match form {
                    None => {}
                    Some(CitationForm::Normal) => normal = true,
                    Some(CitationForm::Prose) => prose = true,
                    Some(_) => {
                        errors.push(error!(
                            child.span(),
                            "citation form is not supported by citation engine \"citum\"",
                        ));
                        continue;
                    }
                }

                hidden &= citation_hidden;
                self.cited.push(key.clone());

                items.push(citum_engine::CitationItem {
                    id: key,
                    suffix: child
                        .supplement
                        .get_cloned(StyleChain::default())
                        .map(|content| content.plain_text().to_string()),
                    ..Default::default()
                });
            }

            if prose && normal {
                errors.push(error!(
                    first.span(),
                    "mixed prose and normal citation forms are not supported by citation engine \"citum\"",
                ));
            }

            if !errors.is_empty() {
                self.failures.insert(location, Err(errors));
                continue;
            }

            self.infos
                .push(CitumGroupInfo { location, span: first.span(), hidden });

            citations.push(citum_engine::Citation {
                id: Some(format!("typst-{index}")),
                mode: if prose {
                    citum_engine::reference::CitationMode::Integral
                } else {
                    citum_engine::reference::CitationMode::NonIntegral
                },
                items,
                ..Default::default()
            });
        }

        citations
    }

    /// Display the Citum citation strings as Typst content.
    fn display_citations(
        &mut self,
        rendered: &[String],
    ) -> FxHashMap<Location, SourceResult<Content>> {
        let mut output = std::mem::take(&mut self.failures);
        for (info, text) in self.infos.iter().zip(rendered) {
            let mut content = if info.hidden {
                Content::empty()
            } else {
                TextElem::packed(text.clone()).spanned(info.span)
            };

            if !info.hidden && self.database.is_note_style() {
                content = FootnoteElem::with_content(content).pack();
            }

            output.insert(info.location, Ok(content));
        }

        output
    }

    /// Display the Citum bibliography entries as Typst content.
    #[allow(clippy::type_complexity)]
    fn display_references(
        &self,
        processor: &citum_engine::Processor,
    ) -> Vec<(Option<Content>, Content, Location)> {
        let full = self.bibliography.full.get(StyleChain::default());
        let ids: Vec<String> = if full {
            self.database.iter_keys().map(str::to_string).collect()
        } else {
            let mut seen = FxHashSet::default();
            self.cited
                .iter()
                .filter(|id| seen.insert((*id).clone()))
                .cloned()
                .collect()
        };

        let rendered = processor
            .render_selected_bibliography_with_format::<
                citum_engine::render::plain::PlainText,
                _,
            >(ids);

        let location = self.bibliography.location().unwrap();
        rendered
            .lines()
            .filter(|line| !line.trim().is_empty())
            .enumerate()
            .map(|(k, line)| {
                (
                    None,
                    TextElem::packed(line.to_owned()).spanned(self.bibliography.span()),
                    location.variant(k + 1),
                )
            })
            .collect()
    }
}

/// Context for generating the bibliography with Hayagriva.
struct HayagrivaGenerator<'a> {
    /// The world that is used to evaluate mathematical material in citations.
    world: Tracked<'a, dyn World + 'a>,
    /// The document's bibliography.
    bibliography: Packed<BibliographyElem>,
    /// The document's citation groups.
    groups: EcoVec<Content>,
    /// Details about each group that are accumulated while driving hayagriva's
    /// bibliography driver and needed when processing hayagriva's output.
    infos: Vec<GroupInfo>,
    /// Citations with unresolved keys.
    failures: FxHashMap<Location, SourceResult<Content>>,
}

/// Details about a group of merged citations. All citations are put into groups
/// of adjacent ones (e.g., `@foo @bar` will merge into a group of length two).
/// Even single citations will be put into groups of length one.
struct GroupInfo {
    /// The group's location.
    location: Location,
    /// The group's span.
    span: Span,
    /// Whether the group should be displayed in a footnote.
    footnote: bool,
    /// Details about the groups citations.
    subinfos: SmallVec<[CiteInfo; 1]>,
}

/// Details about a citation item in a request.
struct CiteInfo {
    /// The citation's key.
    key: Label,
    /// The citation's supplement.
    supplement: Option<Content>,
    /// Whether this citation was hidden.
    hidden: bool,
}

impl<'a> HayagrivaGenerator<'a> {
    /// Create a new generator.
    fn new(
        world: Tracked<'a, dyn World + 'a>,
        bibliography: Packed<BibliographyElem>,
        groups: EcoVec<Content>,
    ) -> StrResult<Self> {
        let infos = Vec::with_capacity(groups.len());
        Ok(Self {
            world,
            bibliography,
            groups,
            infos,
            failures: FxHashMap::default(),
        })
    }

    /// Drives hayagriva's citation driver.
    fn drive(&mut self) -> hayagriva::Rendered {
        static LOCALES: LazyLock<Vec<citationberg::Locale>> =
            LazyLock::new(hayagriva::archive::locales);

        let database = self.bibliography.sources.derived.as_hayagriva();
        let bibliography_style =
            &self.bibliography.style.get_ref(StyleChain::default()).derived;

        // Process all citation groups.
        let mut driver = BibliographyDriver::new();
        for elem in &self.groups {
            let group = elem.to_packed::<CiteGroup>().unwrap();
            let location = elem.location().unwrap();
            let children = &group.children;

            // Groups should never be empty.
            let Some(first) = children.first() else { continue };

            let mut subinfos = SmallVec::with_capacity(children.len());
            let mut items = Vec::with_capacity(children.len());
            let mut errors = EcoVec::new();
            let mut normal = true;

            // Create infos and items for each child in the group.
            for child in children {
                let Some(entry) = database.get(child.key) else {
                    errors.push(error!(
                        child.span(),
                        "key `{}` does not exist in the bibliography",
                        child.key.resolve(),
                    ));
                    continue;
                };

                let supplement = child.supplement.get_cloned(StyleChain::default());
                let locator = supplement.as_ref().map(|c| {
                    SpecificLocator(
                        citationberg::taxonomy::Locator::Custom,
                        hayagriva::LocatorPayload::Transparent(TransparentLocator::new(
                            c.clone(),
                        )),
                    )
                });

                let mut hidden = false;
                let special_form = match child.form.get(StyleChain::default()) {
                    None => {
                        hidden = true;
                        None
                    }
                    Some(CitationForm::Normal) => None,
                    Some(CitationForm::Prose) => Some(hayagriva::CitePurpose::Prose),
                    Some(CitationForm::Full) => Some(hayagriva::CitePurpose::Full),
                    Some(CitationForm::Author) => Some(hayagriva::CitePurpose::Author),
                    Some(CitationForm::Year) => Some(hayagriva::CitePurpose::Year),
                };

                normal &= special_form.is_none();
                subinfos.push(CiteInfo { key: child.key, supplement, hidden });
                items.push(CitationItem::new(entry, locator, None, hidden, special_form));
            }

            if !errors.is_empty() {
                self.failures.insert(location, Err(errors));
                continue;
            }

            let style = match first.style.get_ref(StyleChain::default()) {
                Smart::Auto => bibliography_style.get(),
                Smart::Custom(style) => style.derived.get(),
            };

            self.infos.push(GroupInfo {
                location,
                subinfos,
                span: first.span(),
                footnote: normal
                    && style.settings.class == citationberg::StyleClass::Note,
            });

            driver.citation(CitationRequest::new(
                items,
                style,
                Some(locale(first.lang.unwrap_or(Lang::ENGLISH), first.region.flatten())),
                &LOCALES,
                None,
            ));
        }

        let locale = locale(
            self.bibliography.lang.unwrap_or(Lang::ENGLISH),
            self.bibliography.region.flatten(),
        );

        // Add hidden items for everything if we should print the whole
        // bibliography.
        if self.bibliography.full.get(StyleChain::default()) {
            for (_, entry) in database.iter() {
                driver.citation(CitationRequest::new(
                    vec![CitationItem::new(entry, None, None, true, None)],
                    bibliography_style.get(),
                    Some(locale.clone()),
                    &LOCALES,
                    None,
                ));
            }
        }

        driver.finish(BibliographyRequest {
            style: bibliography_style.get(),
            locale: Some(locale),
            locale_files: &LOCALES,
        })
    }

    /// Displays hayagriva's output as content for the citations and references.
    fn display(&mut self, rendered: &hayagriva::Rendered) -> StrResult<Works> {
        let citations = self.display_citations(rendered)?;
        let references = self.display_references(rendered)?;
        let hanging_indent =
            rendered.bibliography.as_ref().is_some_and(|b| b.hanging_indent);
        Ok(Works { citations, references, hanging_indent })
    }

    /// Display the citation groups.
    fn display_citations(
        &mut self,
        rendered: &hayagriva::Rendered,
    ) -> StrResult<FxHashMap<Location, SourceResult<Content>>> {
        // Determine for each citation key where in the bibliography it is,
        // so that we can link there.
        let mut links = FxHashMap::default();
        if let Some(bibliography) = &rendered.bibliography {
            let location = self.bibliography.location().unwrap();
            for (k, item) in bibliography.items.iter().enumerate() {
                links.insert(item.key.as_str(), location.variant(k + 1));
            }
        }

        let mut output = std::mem::take(&mut self.failures);
        for (info, citation) in self.infos.iter().zip(&rendered.citations) {
            let supplement = |i: usize| info.subinfos.get(i)?.supplement.clone();
            let link = |i: usize| {
                links.get(info.subinfos.get(i)?.key.resolve().as_str()).copied()
            };

            let renderer = ElemRenderer {
                world: self.world,
                span: info.span,
                supplement: &supplement,
                link: &link,
            };

            let content = if info.subinfos.iter().all(|sub| sub.hidden) {
                Content::empty()
            } else {
                let mut content =
                    renderer.display_elem_children(&citation.citation, None, true)?;

                if info.footnote {
                    content = FootnoteElem::with_content(content).pack();
                }

                content
            };

            output.insert(info.location, Ok(content));
        }

        Ok(output)
    }

    /// Display the bibliography references.
    #[allow(clippy::type_complexity)]
    fn display_references(
        &self,
        rendered: &hayagriva::Rendered,
    ) -> StrResult<Option<Vec<(Option<Content>, Content, Location)>>> {
        let Some(rendered) = &rendered.bibliography else { return Ok(None) };

        // Determine for each citation key where it first occurred, so that we
        // can link there.
        let mut first_occurrences = FxHashMap::default();
        for info in &self.infos {
            for subinfo in &info.subinfos {
                let key = subinfo.key.resolve();
                first_occurrences.entry(key).or_insert(info.location);
            }
        }

        // The location of the bibliography.
        let location = self.bibliography.location().unwrap();

        let mut output = vec![];
        for (k, item) in rendered.items.iter().enumerate() {
            let renderer = ElemRenderer {
                world: self.world,
                span: self.bibliography.span(),
                supplement: &|_| None,
                link: &|_| None,
            };

            // Each reference is assigned a manually created well-known location
            // that is derived from the bibliography's location. This way,
            // citations can link to them.
            let backlink = location.variant(k + 1);

            // Render the first field.
            let mut prefix = item
                .first_field
                .as_ref()
                .map(|elem| renderer.display_elem_child(elem, None, false))
                .transpose()?;

            // Render the main reference content.
            let reference = renderer.display_elem_children(
                &item.content,
                Some(&mut prefix),
                false,
            )?;

            let prefix = prefix.map(|content| {
                if let Some(location) = first_occurrences.get(item.key.as_str()) {
                    let alt = content.plain_text();
                    let body = content.spanned(self.bibliography.span());
                    DirectLinkElem::new(*location, body, Some(alt)).pack()
                } else {
                    content
                }
            });

            output.push((prefix, reference, backlink));
        }

        Ok(Some(output))
    }
}

/// Renders hayagriva elements into content.
struct ElemRenderer<'a> {
    /// The world that is used to evaluate mathematical material.
    world: Tracked<'a, dyn World + 'a>,
    /// The span that is attached to all of the resulting content.
    span: Span,
    /// Resolves the supplement of i-th citation in the request.
    supplement: &'a dyn Fn(usize) -> Option<Content>,
    /// Resolves where the i-th citation in the request should link to.
    link: &'a dyn Fn(usize) -> Option<Location>,
}

impl ElemRenderer<'_> {
    /// Display rendered hayagriva elements.
    ///
    /// The `prefix` can be a separate content storage where `left-margin`
    /// elements will be accumulated into.
    ///
    /// `is_citation` dictates whether whitespace at the start of the citation
    /// will be eliminated. Some CSL styles yield whitespace at the start of
    /// their citations, which should instead be handled by Typst.
    fn display_elem_children(
        &self,
        elems: &hayagriva::ElemChildren,
        mut prefix: Option<&mut Option<Content>>,
        is_citation: bool,
    ) -> StrResult<Content> {
        Ok(Content::sequence(
            elems
                .0
                .iter()
                .enumerate()
                .map(|(i, elem)| {
                    self.display_elem_child(
                        elem,
                        prefix.as_deref_mut(),
                        is_citation && i == 0,
                    )
                })
                .collect::<StrResult<Vec<_>>>()?,
        ))
    }

    /// Display a rendered hayagriva element.
    fn display_elem_child(
        &self,
        elem: &hayagriva::ElemChild,
        prefix: Option<&mut Option<Content>>,
        trim_start: bool,
    ) -> StrResult<Content> {
        Ok(match elem {
            hayagriva::ElemChild::Text(formatted) => {
                self.display_formatted(formatted, trim_start)
            }
            hayagriva::ElemChild::Elem(elem) => self.display_elem(elem, prefix)?,
            hayagriva::ElemChild::Markup(markup) => self.display_math(markup),
            hayagriva::ElemChild::Link { text, url } => self.display_link(text, url)?,
            hayagriva::ElemChild::Transparent { cite_idx, format } => {
                self.display_transparent(*cite_idx, format)
            }
        })
    }

    /// Display a block-level element.
    fn display_elem(
        &self,
        elem: &hayagriva::Elem,
        mut prefix: Option<&mut Option<Content>>,
    ) -> StrResult<Content> {
        use citationberg::Display;

        let block_level = matches!(elem.display, Some(Display::Block | Display::Indent));

        let mut content = self.display_elem_children(
            &elem.children,
            if block_level { None } else { prefix.as_deref_mut() },
            false,
        )?;

        match elem.display {
            Some(Display::Block) => {
                content = BlockElem::packed(content).spanned(self.span);
            }
            Some(Display::Indent) => {
                content = CslIndentElem::new(content).pack().spanned(self.span);
            }
            Some(Display::LeftMargin) => {
                // The `display="left-margin"` attribute is only supported at
                // the top-level (when prefix is `Some(_)`). Within a
                // block-level container, it is ignored. The CSL spec is not
                // specific about this, but it is in line with citeproc.js's
                // behaviour.
                if let Some(prefix) = prefix {
                    *prefix.get_or_insert_with(Default::default) += content;
                    return Ok(Content::empty());
                }
            }
            _ => {}
        }

        content = content.spanned(self.span);

        if let Some(hayagriva::ElemMeta::Entry(i)) = elem.meta
            && let Some(location) = (self.link)(i)
        {
            let alt = content.plain_text();
            content = DirectLinkElem::new(location, content, Some(alt)).pack();
        }

        Ok(content)
    }

    /// Display math.
    fn display_math(&self, math: &str) -> Content {
        let library = self.world.library();
        (library.routines.eval_string)(
            self.world,
            library,
            // TODO: propagate warnings
            Sink::new().track_mut(),
            EmptyIntrospector.track(),
            Context::none().track(),
            math,
            SpanMode::Uniform(self.span),
            SyntaxMode::Math,
            Scope::new(),
        )
        .map(Value::display)
        .unwrap_or_else(|_| TextElem::packed(math).spanned(self.span))
    }

    /// Display a link.
    fn display_link(&self, text: &hayagriva::Formatted, url: &str) -> StrResult<Content> {
        let dest = Destination::Url(Url::new(url)?);
        Ok(LinkElem::new(dest.into(), self.display_formatted(text, false))
            .pack()
            .spanned(self.span))
    }

    /// Display transparent pass-through content.
    fn display_transparent(&self, i: usize, format: &hayagriva::Formatting) -> Content {
        let content = (self.supplement)(i).unwrap_or_default();
        apply_formatting(content, format)
    }

    /// Display formatted hayagriva text as content.
    fn display_formatted(
        &self,
        formatted: &hayagriva::Formatted,
        trim_start: bool,
    ) -> Content {
        let formatted_text = if trim_start {
            formatted.text.trim_start()
        } else {
            formatted.text.as_str()
        };

        let content = TextElem::packed(formatted_text).spanned(self.span);
        apply_formatting(content, &formatted.formatting)
    }
}

/// Applies formatting to content.
fn apply_formatting(mut content: Content, format: &hayagriva::Formatting) -> Content {
    match format.font_style {
        citationberg::FontStyle::Normal => {}
        citationberg::FontStyle::Italic => {
            content = content.emph();
        }
    }

    match format.font_variant {
        citationberg::FontVariant::Normal => {}
        citationberg::FontVariant::SmallCaps => {
            content = SmallcapsElem::new(content).pack();
        }
    }

    match format.font_weight {
        citationberg::FontWeight::Normal => {}
        citationberg::FontWeight::Bold => {
            content = content.strong();
        }
        citationberg::FontWeight::Light => {
            // We don't have a semantic element for "light" and a `StrongElem`
            // with negative delta does not have the appropriate semantics, so
            // keeping this as a direct style.
            content = CslLightElem::new(content).pack();
        }
    }

    match format.text_decoration {
        citationberg::TextDecoration::None => {}
        citationberg::TextDecoration::Underline => {
            content = content.underlined();
        }
    }

    let span = content.span();
    match format.vertical_align {
        citationberg::VerticalAlign::None => {}
        citationberg::VerticalAlign::Baseline => {}
        citationberg::VerticalAlign::Sup => {
            // Add zero-width weak spacing to make the superscript "sticky".
            content =
                HElem::hole().clone() + SuperElem::new(content).pack().spanned(span);
        }
        citationberg::VerticalAlign::Sub => {
            content = HElem::hole().clone() + SubElem::new(content).pack().spanned(span);
        }
    }

    content
}

/// Create a locale code from language and optionally region.
fn locale(lang: Lang, region: Option<Region>) -> citationberg::LocaleCode {
    let mut value = String::with_capacity(5);
    value.push_str(lang.as_str());
    if let Some(region) = region {
        value.push('-');
        value.push_str(region.as_str())
    }
    citationberg::LocaleCode(value)
}

/// Translation of `font-weight="light"` in CSL.
///
/// We translate `font-weight: "bold"` to `<strong>` since it's likely that the
/// CSL spec just talks about bold because it has no notion of semantic
/// elements. The benefits of a strict reading of the spec are also rather
/// questionable, while using semantic elements makes the bibliography more
/// accessible, easier to style, and more portable across export targets.
#[elem]
pub struct CslLightElem {
    #[required]
    pub body: Content,
}

/// Translation of `display="indent"` in CSL.
///
/// A `display="block"` is simply translated to a Typst `BlockElem`. Similarly,
/// we could translate `display="indent"` to a `PadElem`, but (a) it does not
/// yet have support in HTML and (b) a `PadElem` described a fixed padding while
/// CSL leaves the amount of padding user-defined so it's not a perfect fit.
#[elem]
pub struct CslIndentElem {
    #[required]
    pub body: Content,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bibliography_load_builtin_styles() {
        for &archived in ArchivedStyle::all() {
            let _ = CslStyle::from_archived(archived);
        }
    }

    #[test]
    fn test_csl_source_cast_info_include_all_names() {
        let CastInfo::Union(cast_info) = CslSource::input() else {
            panic!("the cast info of CslSource should be a union");
        };

        let missing: Vec<_> = ArchivedStyle::all()
            .iter()
            .flat_map(|style| style.names())
            .filter(|name| {
                let found = cast_info.iter().any(|info| match info {
                    CastInfo::Value(Value::Str(n), _) => n.as_str() == **name,
                    _ => false,
                });
                !found
            })
            .collect();

        assert!(
            missing.is_empty(),
            "missing style names in CslSource cast info: '{missing:?}'"
        );
    }
}
