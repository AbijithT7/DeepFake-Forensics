import React, { useEffect, useRef, useState } from 'react';
import { Scan, Sun, Moon, Menu, X, ArrowUpRight } from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';

export function Header() {
  const { activePage, navigateTo, theme, toggleTheme } = useAnalysis();
  const [open, setOpen] = useState(false);
  const menuButton = useRef(null);
  useEffect(() => setOpen(false), [activePage]);
  useEffect(() => {
    const close = event => { if (event.key === 'Escape') { setOpen(false); menuButton.current?.focus(); } };
    if (open) window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, [open]);
  const nav = [['landing', 'Home'], ['analyze', 'Analyze'], ['history', 'Reports'], ['about', 'About']];
  const current = ['results', 'explainability', 'indicators', 'report'].includes(activePage) ? 'analyze' : activePage;
  return <header className="site-header">
    <div className="container header-inner">
      <a className="wordmark" href="#/" aria-label="DeepTrace home"><Scan aria-hidden="true" /><span>DeepTrace<span className="brand-period">.</span></span></a>
      <nav className="desktop-nav" aria-label="Main navigation">
        {nav.map(([page, label]) => <a key={page} href={page === 'landing' ? '#/' : `#/${page}`} aria-current={current === page ? 'page' : undefined}>{label}</a>)}
      </nav>
      <div className="header-actions">
        <button className="icon-button theme-toggle" onClick={toggleTheme} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>{theme === 'dark' ? <Sun /> : <Moon />}</button>
        <button className="button button-small header-cta" onClick={() => navigateTo('analyze')}>Analyze image <ArrowUpRight size={17} /></button>
        <button ref={menuButton} className="icon-button menu-toggle" aria-label={open ? 'Close menu' : 'Open menu'} aria-expanded={open} aria-controls="mobile-navigation" onClick={() => setOpen(value => !value)}>{open ? <X /> : <Menu />}</button>
      </div>
    </div>
    {open && <nav id="mobile-navigation" className="mobile-nav" aria-label="Mobile navigation">{nav.map(([page, label]) => <a key={page} href={page === 'landing' ? '#/' : `#/${page}`} aria-current={current === page ? 'page' : undefined} onClick={() => setOpen(false)}>{label}<ArrowUpRight size={18} /></a>)}</nav>}
  </header>;
}
