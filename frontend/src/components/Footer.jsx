import React from 'react';
import { Scan, ArrowUpRight } from 'lucide-react';
export function Footer() {
  return <footer className="site-footer"><div className="container footer-inner">
    <a href="#/" className="wordmark"><Scan aria-hidden="true" /><span>DeepTrace.</span></a>
    <p>A closer look at what’s real.</p>
    <a className="text-link" href="#/about">About the project <ArrowUpRight size={16} /></a>
  </div></footer>;
}
