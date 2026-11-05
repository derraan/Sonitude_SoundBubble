export function EmptyState({
  title,
  body,
}: {
  title: string;
  body: string;
}) {
  return (
    <div className="card-mist px-6 py-10 text-center">
      <p className="font-display text-[22px] tracking-[-0.2px] text-ink-black">{title}</p>
      <p className="mx-auto mt-2 max-w-md text-[15px] leading-relaxed text-slate-gray">{body}</p>
    </div>
  );
}
