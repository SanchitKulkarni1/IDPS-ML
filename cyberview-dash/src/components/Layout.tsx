import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Radio,
  BarChart3,
  Lightbulb,
  FileText,
  Users,
  Activity,
  ShieldBan,
} from "lucide-react";

const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Live Detection", href: "/live-detection", icon: Radio },
  { name: "Live Analysis", href: "/live-analysis", icon: Activity },
  { name: "Traffic Analytics", href: "/traffic-analytics", icon: BarChart3 },
  // { name: "Explainability", href: "/explainability", icon: Lightbulb },
  { name: "Prevention", href: "/prevention", icon: ShieldBan },
  { name: "Logs & Reports", href: "/logs", icon: FileText },
  { name: "About Project", href: "/about", icon: Users },
];

interface LayoutProps {
  children: React.ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  return (
    <div className="flex bg-secondary/30 w-[800px] h-[600px] overflow-hidden">
      {/* Sidebar */}
      <aside className="w-48 bg-card border-r border-border flex flex-col">
        <div className="p-4 border-b border-border">
          <h1 className="text-xl font-bold text-primary">CyberView</h1>
          <p className="text-[10px] text-muted-foreground mt-1">Monitor & Prevent</p>
        </div>

        <nav className="p-3 space-y-1.5 overflow-y-auto flex-1">
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              end={item.href === "/"}
              className={({ isActive }) =>
                isActive ? "nav-link nav-link-active !py-2" : "nav-link nav-link-inactive !py-2"
              }
            >
              <item.icon className="h-4 w-4 shrink-0" />
              <span className="text-sm truncate">{item.name}</span>
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <div className="p-4 md:p-6">
          {children}
        </div>
      </main>
    </div>
  );
};

export default Layout;
