import type { ReactNode } from "react";

export function PageHeader({
  title,
  children,
  actions,
}: {
  title: string;
  children?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-8 flex flex-wrap items-end justify-between gap-6">
      <div className="max-w-2xl">
        <h2 className="font-display text-[44px] leading-[1.2] tracking-[-0.66px] text-ink-black">
          {title}
        </h2>
        {children ? (
          <div className="mt-3 text-[17px] leading-[1.45] text-slate-gray">{children}</div>
        ) : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-3">{actions}</div> : null}
    </div>
  );
}
